"""Capture actual runtime capabilities without installing/changing CUDA."""
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    report={"python":sys.version,"executable":sys.executable,"platform":platform.platform()}
    result=subprocess.run(["nvidia-smi"],capture_output=True,text=True,errors="replace")
    report["nvidia_smi"]=result.stdout
    if importlib.util.find_spec("torch"):
        import torch
        report["torch"]={"version":torch.__version__,"cuda_runtime":torch.version.cuda,
                         "cuda_available":torch.cuda.is_available()}
    else:
        report["torch"]={"status":"NOT_INSTALLED","cuda_runtime":None,"cuda_available":None}
    if importlib.util.find_spec("tensorflow"):
        import tensorflow as tf
        report["tensorflow"]={"version":tf.__version__,"gpus":[str(x) for x in tf.config.list_physical_devices("GPU")]}
    folder=Path(__file__).resolve().parent/"reports";folder.mkdir(exist_ok=True)
    (folder/"training_environment.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()