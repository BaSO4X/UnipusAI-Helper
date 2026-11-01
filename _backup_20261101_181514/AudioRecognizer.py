"""本地语音识别（faster-whisper，GPU 专用）。

只在 CUDA 上运行。没有可用显卡、或 CUDA 运行库缺失时直接抛错，
不做 CPU 回退——避免在用户不知情的情况下用慢速 CPU 跑完整个转写。

两个环境坑在这里一次解决：
1. faster-whisper 用 av 解码音频，但新版 av 移除了 metadata_errors 参数，
   与 faster-whisper 的调用不兼容；这里改用系统 ffmpeg 解码成 float32 PCM。
2. pip 装的 nvidia-cublas-cu12 / nvidia-cudnn-cu12 的 DLL 不在系统 PATH 里，
   导致 "cublas64_12.dll is not found"；这里在加载模型前把目录塞进 PATH。
"""
import os
import subprocess
import tempfile
import traceback
import urllib.error
import urllib.request
from shutil import which
from typing import Dict, List, Optional

# Whisper 固定要求 16k 单声道
SAMPLE_RATE = 16000


class CudaUnavailableError(RuntimeError):
    """没有可用的 CUDA 环境。"""


def _ensure_cuda_dll_paths() -> None:
    """把 pip 安装的 nvidia CUDA 运行库目录加入 DLL 搜索路径。

    nvidia-cublas-cu12 / nvidia-cudnn-cu12 的 DLL 位于 site-packages/nvidia/*/bin，
    不在系统 PATH 中，ctranslate2 会报 cublas64_12.dll 找不到。
    """
    if os.name != 'nt':
        return
    try:
        import site
        roots: List[str] = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
    except Exception:
        roots = []

    added: List[str] = []
    for root in roots:
        nv_root = os.path.join(root, 'nvidia')
        if not os.path.isdir(nv_root):
            continue
        for package in os.listdir(nv_root):
            bin_dir = os.path.join(nv_root, package, 'bin')
            if os.path.isdir(bin_dir) and bin_dir not in added:
                added.append(bin_dir)

    if not added:
        return

    current = os.environ.get('PATH', '')
    for path in added:
        if path not in current:
            current = path + os.pathsep + current
    os.environ['PATH'] = current

    for path in added:
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):
            pass


def _find_ffmpeg() -> Optional[str]:
    """定位 ffmpeg 可执行文件。"""
    found = which('ffmpeg')
    if found:
        return found
    for candidate in (
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        os.path.expandvars(r'%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe'),
        os.path.expandvars(r'%USERPROFILE%\ffmpeg\bin\ffmpeg.exe'),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _decode_with_ffmpeg(audio_path: str):
    """用系统 ffmpeg 把音频解码成 16k 单声道 float32 numpy 数组。"""
    import numpy as np

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise CudaUnavailableError(
            "未找到 ffmpeg。请安装 FFmpeg 并加入 PATH（ffmpeg -version 验证）。")

    completed = subprocess.run(
        [ffmpeg, '-nostdin', '-v', 'error', '-i', audio_path,
         '-f', 'f32le', '-acodec', 'pcm_f32le',
         '-ac', '1', '-ar', str(SAMPLE_RATE), '-'],
        capture_output=True, check=True, timeout=300,
    )
    if not completed.stdout:
        return None
    return np.frombuffer(completed.stdout, dtype=np.float32)


class AudioTranscriber:
    """本地 Whisper 语音识别，仅使用 CUDA（显卡）。

    模型采用懒加载：构造时不碰 CUDA，只有真正要转写时才初始化。
    这样程序启动不会因为显卡/显存/CUDA 环境问题直接崩掉。
    """

    def __init__(self, config=None):
        self.config = config
        self.local_model = None
        self.device = 'cuda'
        self.compute_type = 'float16'
        self.model_size = 'base'
        self._transcript_cache: Dict[str, str] = {}
        self._init_failed = False
        self._init_error: Optional[str] = None
        self._read_config(config)

    def ensure_ready(self) -> bool:
        """确保模型已加载。返回是否可用；失败只记录，不抛异常。

        注意：CUDA 初始化出问题时可能直接终止进程（C++ 层崩溃，Python 抓不到），
        因此这里先做前置体检，通过后才真正加载模型。
        """
        if self.local_model is not None:
            return True
        if self._init_failed:
            return False

        reason = self._precheck()
        if reason:
            self._init_failed = True
            self._init_error = reason
            print(f"       语音识别已禁用：{reason}")
            return False

        try:
            self._init_local_model()
            return self.local_model is not None
        except Exception as exc:
            self._init_failed = True
            self._init_error = str(exc)
            print(f"       语音识别不可用（{self._init_error.splitlines()[0][:100]}）")
            if self._init_error.count('\n'):
                for line in self._init_error.splitlines()[1:]:
                    print(f"         {line.strip()}")
            return False

    def _precheck(self) -> Optional[str]:
        """加载前体检：依赖、显卡、显存、DLL、模型文件。有问题返回原因。"""
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return "未安装 faster-whisper"
        try:
            import ctranslate2
        except ImportError:
            return "未安装 ctranslate2"

        try:
            if ctranslate2.get_cuda_device_count() <= 0:
                return "未检测到可用的 NVIDIA 显卡"
        except Exception as exc:
            return f"CUDA 设备查询失败: {str(exc)[:80]}"

        # 显存检查：至少留 2GB 余量
        free_mb = self._free_vram_mb()
        if free_mb is not None and free_mb < 2048:
            return f"显存不足（可用 {free_mb} MB，至少需要 2048 MB）"

        # CUDA 运行库检查（缺失会让进程直接崩，而不是抛异常）
        if os.name == 'nt':
            _ensure_cuda_dll_paths()
            for dll in ('cublas64_12.dll', 'cublasLt64_12.dll'):
                if not self._find_dll(dll):
                    return f"缺少 CUDA 运行库 {dll}，请运行: pip install nvidia-cublas-cu12"

        # 模型文件检查：0 字节或缺失都会在加载时崩
        size = self._cached_model_size()
        if size is not None and size < 1000000:
            return f"语音模型文件不完整（{size} 字节），请删除缓存后重新下载"
        return None

    @staticmethod
    def _find_dll(name: str) -> bool:
        """在 PATH 与 nvidia 包目录里找 DLL。"""
        import glob as _glob
        if which(name):
            return True
        try:
            import site
            roots = list(site.getsitepackages())
            user_site = site.getusersitepackages()
            if user_site:
                roots.append(user_site)
        except Exception:
            roots = []
        for root in roots:
            pattern = os.path.join(root, 'nvidia', '*', 'bin', name)
            if _glob.glob(pattern):
                return True
        try:
            return bool(_glob.glob(os.path.join(os.environ.get('WINDIR', r'C:\Windows'),
                                                'System32', name)))
        except Exception:
            return False

    @staticmethod
    def _free_vram_mb() -> Optional[int]:
        """查询可用显存（MB）。拿不到返回 None。"""
        try:
            completed = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in (completed.stdout or '').splitlines() if l.strip()]
            return int(lines[0]) if lines else None
        except Exception:
            return None

    def _cached_model_size(self) -> Optional[int]:
        """查本地已缓存的模型文件大小；未缓存返回 None。"""
        try:
            from huggingface_hub import scan_cache_dir
            for repo in scan_cache_dir().repos:
                if self.model_size in repo.repo_id.lower():
                    total = 0
                    for revision in repo.revisions:
                        for f in revision.files:
                            if f.file_name.endswith('.bin'):
                                total += f.size_on_disk
                    return total
        except Exception:
            pass
        return None

    # ---- 配置 ---------------------------------------------------------
    def _read_config(self, config) -> None:
        def get(key, default):
            if config is None:
                return default
            return getattr(config, key, default)

        self.model_size = (get('whisper_model', 'base') or 'base').strip()
        self.compute_type = (get('whisper_compute_type', 'auto') or 'auto').strip()

    @staticmethod
    def _print_exception(prefix: str, exc: Exception) -> None:
        """完整输出异常信息与堆栈，方便定位问题"""
        print(f"{prefix}: {exc}")
        tb = traceback.format_exc().rstrip()
        if tb and tb != "NoneType: None":
            print(tb)

    # ---- 模型初始化 ---------------------------------------------------
    def _init_local_model(self):
        """初始化本地模型；CUDA 不可用时抛错，不回退 CPU。"""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise CudaUnavailableError(
                "未安装 faster-whisper，请运行: pip install faster-whisper"
            ) from exc

        _ensure_cuda_dll_paths()

        try:
            import ctranslate2
        except ImportError as exc:
            raise CudaUnavailableError(
                "未安装 ctranslate2，请运行: pip install ctranslate2"
            ) from exc

        device_count = ctranslate2.get_cuda_device_count()
        if device_count <= 0:
            raise CudaUnavailableError(
                "未检测到可用的 NVIDIA 显卡（CUDA）。\n"
                "  请确认：显卡驱动已安装，且 nvidia-smi 能正常输出。")

        compute = self.compute_type
        if compute == 'auto':
            compute = self._pick_compute_type(ctranslate2)
        self.compute_type = compute

        gpu_name = self._gpu_name()
        label = f"{gpu_name} / {compute}" if gpu_name else compute
        print(f"       加载 Whisper 模型 ({self.model_size}, CUDA: {label})...")

        try:
            self.local_model = WhisperModel(
                self.model_size, device='cuda', compute_type=compute)
        except Exception as exc:
            self._print_exception("       加载本地模型失败", exc)
            raise CudaUnavailableError(
                f"CUDA 模型加载失败: {exc}\n"
                "  常见原因：显存不足、CUDA 运行库缺失（cublas64_12.dll）。"
            ) from exc

        self.device = 'cuda'
        print(f"       本地模型加载完成（CUDA / {compute}）")
        self._selftest()

    @staticmethod
    def _gpu_name() -> str:
        """读显卡型号，仅用于日志。"""
        try:
            completed = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
            )
            line = (completed.stdout or '').strip().splitlines()
            return line[0].strip() if line else ''
        except Exception:
            return ''

    @staticmethod
    def _pick_compute_type(ctranslate2) -> str:
        """按显卡支持能力挑计算精度，优先 float16。"""
        try:
            supported = ctranslate2.get_supported_compute_types('cuda')
        except Exception:
            return 'float16'
        for candidate in ('float16', 'int8_float16', 'int8_float32', 'float32'):
            if candidate in supported:
                return candidate
        return 'float16'

    def _selftest(self) -> None:
        """跑一次 1 秒静音，确认 DLL 齐全、kernel 能编译。

        cublas 缺失这类问题要到第一次推理才暴露，这里提前踩掉。
        """
        try:
            import numpy as np
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            list(self.local_model.transcribe(silence, language='en')[0])
        except Exception as exc:
            print(f"       CUDA 自检失败: {str(exc)[:160]}")
            raise CudaUnavailableError(
                f"CUDA 推理自检失败: {exc}\n"
                "  请运行: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
            ) from exc

    # ---- 对外接口 -----------------------------------------------------
    def transcribe(self, audio_url: str, language: str = "en") -> str:
        """下载音频并转录为文字。模型不可用时返回空串。"""
        if audio_url in self._transcript_cache:
            print(f"       使用缓存的识别结果")
            return self._transcript_cache[audio_url]

        if not self.ensure_ready():
            return ""

        audio_path = None
        try:
            print(f"      【下载】获取音频...")
            with urllib.request.urlopen(audio_url, timeout=30) as response, \
                    tempfile.NamedTemporaryFile(suffix=self._url_suffix(audio_url),
                                                delete=False) as f:
                f.write(response.read())
                audio_path = f.name

            print(f"        开始识别（CUDA）...")
            text = self._transcribe_local(audio_path, language)

            if text:
                self._transcript_cache[audio_url] = text
                print(f"       识别成功 ({len(text)} 字符)")
                print(f"       音频识别结果{text}")

            return text or ""

        except urllib.error.URLError as e:
            self._print_exception("       下载音频失败", e)
            return ""
        except Exception as e:
            # 转写失败不阻断主流程，只记录
            self._print_exception("       识别失败", e)
            return ""
        finally:
            if audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    @staticmethod
    def _url_suffix(url: str) -> str:
        lowered = (url or '').lower().split('?')[0]
        for ext in ('.mp3', '.m4a', '.wav', '.aac', '.ogg', '.flac', '.mp4'):
            if lowered.endswith(ext):
                return ext
        return '.mp3'

    def _transcribe_local(self, audio_path: str, language: str) -> Optional[str]:
        """使用 CUDA 模型识别。"""
        if self.local_model is None and not self.ensure_ready():
            return None

        audio = _decode_with_ffmpeg(audio_path)

        segments, _info = self.local_model.transcribe(
            audio,
            language=language,
            beam_size=5,
            # VAD 需要 onnxruntime（Python 3.15 暂无 wheel），改用模型自带的
            # 无语音判定 + 温度回退，效果足够且少一个重依赖。
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return ''.join(segment.text for segment in segments).strip()
