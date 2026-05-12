from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import pty
import select
import sys
import struct
import subprocess
import fcntl
import termios

def run_with_live_output(cmd):
    """运行命令并实时显示输出，正确处理 tqdm"""
    def get_terminal_size():
        try:
            ws = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, b'\x00' * 8)
            rows, cols = struct.unpack('HHHH', ws)[:2]
            if rows > 0 and cols > 0:
                return rows, cols
        except OSError:
            pass
        return 24, 80
    
    print(f"Running command: \n{cmd}")
    master_fd, slave_fd = pty.openpty()
    # 初始化 pty 终端大小（同步外层终端）
    rows, cols = get_terminal_size()
    winsize = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    output_lines = []

    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    data = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                    if data:
                        print(data, end='', flush=True)
                        output_lines.append(data)
                except OSError:
                    break

            if process.poll() is not None:
                while True:
                    try:
                        data = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                        if not data:
                            break
                        print(data, end='', flush=True)
                        output_lines.append(data)
                    except OSError:
                        break
                break
    finally:
        os.close(master_fd)
    process.wait() 
    return process.returncode, ''.join(output_lines)


def get_train_cmd(input, output, experiment="DRK-GS", image_dir=None, kernel_density="dense"):

    # 路径配置
    image_root = f"/home/matt/cviss/Matt/Dataset/{input}"  # Blendswap/Render/pick/13078_toad
    output_base_dir = f"/home/matt/cviss/Matt/GS-Output"
    output_full_dir = f"{output_base_dir}/{experiment}/{output}"  # Blendswap/Render/pick/13078_toad'strategy}/{output}"  # pick/13078_toad


    cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python train.py "
        f"--gs_type DRK "
        f"--is_unbounded "
        f"--kernel_density {kernel_density} " # f"--cache_sort "
        f"-s {image_root} "
        f"-m {output_full_dir} "
        f"{f'-i {image_dir} ' if image_dir is not None else ''}"
        f"-r 1 "
        f"--eval "
        f"--iterations 30000 "
        f"--save_iterations 7000 30000 "

    )
    # ============================================================
    # 评估命令
    # ============================================================
    eval_cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python evaluate_metrics.py -s {image_root} -m {output_full_dir} --iterations 7000 30000 "
        f"--save_image "
        # f"--skip_train"
    )
    # eval_cmd = None
    return cmd, eval_cmd

@dataclass
class Experiment:
    kernel_density: str
    subtitle: str = ""
    


if __name__ == "__main__":
    # for factor, strategy in product([2], ["mcmc", "default"]):
    #     # 示例用法
    #     cmd, eval_cmd = get_train_cmd(input="Rogers/Tower_0529", output=f"Rogers/Tower_0529_{factor}", factor=factor, strategy=strategy)

    #     run_with_live_output(cmd)
    #     run_with_live_output(eval_cmd)
    
    
    
    # MipNerf-360 dataset
    dataset_root = Path("/mnt/cviss/Matt/Dataset")
    mip_root = dataset_root / "Mip-NeRF360/360_v2/"
    tnt_root = dataset_root / "tanks_and_temples"
    gs_output_root = Path("/mnt/cviss/Matt/GS-Output")

    scenes = [("Rogers/Tower_0529", "Rogers/Tower_0529")]
    scenes = []
    scenes.extend([
        (f"Mip-NeRF360/360_v2/{d.name}", f"Mip-NeRF360/360_v2/{d.name}")
        for d in mip_root.iterdir() if d.is_dir()
    ])
    scenes = []
    scenes.extend([
        (f"tanks_and_temples/{d.name}", f"tanks_and_temples/{d.name}")
        for d in tnt_root.iterdir() if d.is_dir()
    ])
    factors = [1]
    experiments = [
        Experiment(kernel_density="dense"),
        Experiment(kernel_density="middle", subtitle="_S2"),
        Experiment(kernel_density="sparse", subtitle="_S1"),
    ]
    parameters = list(product(experiments, scenes, factors))
    total_runs = len(parameters)
    for i, (exp, scene, factor) in enumerate(parameters):
        print(f"Running experiment {i+1}/{total_runs}: {exp}, scene={scene}, factor={factor}")
        input_path, output_base = scene
        output_path = f"{output_base}_{factor}" if factor != 1 else output_base
        image_dir=f"images_{factor}" if factor != 1 else "images"
        scene_dir = f"{input_path}"
        
        trained_file = gs_output_root / f"DRK-GS{exp.subtitle}" / output_path / "training_time.json"
        eval_file = gs_output_root / f"DRK-GS{exp.subtitle}" / output_path / "results_iter30000.json"
        
        trained_flag = trained_file.exists()
        eval_flag = eval_file.exists()
        if trained_flag and eval_flag:
            continue
        
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
            if "train" not in eval_data or "test" not in eval_data:
                eval_flag = False
        
        cmd, eval_cmd = get_train_cmd(
            input=scene_dir,
            output=output_path,
            experiment=f"DRK-GS{exp.subtitle}",
            image_dir=image_dir,
            kernel_density=exp.kernel_density
        )
        if not trained_flag:
            run_with_live_output(cmd)
        if not eval_flag:
            run_with_live_output(eval_cmd)