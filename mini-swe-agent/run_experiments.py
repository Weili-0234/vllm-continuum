#!/usr/bin/env python3
"""
实验脚本：自动运行不同scheduling_policy和jps参数组合的实验
"""
import subprocess
import time
import os
import signal
import sys
from pathlib import Path
from typing import List, Dict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置区域 ====================
# 定义要测试的参数值
SCHEDULING_POLICIES = ["fcfs", "continuum"]  # 修改为你需要测试的scheduling_policy值
JPS_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]  # 修改为你需要测试的jps值

# 其他配置
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
PORT_1 = 8001
PORT_2 = 8002
ROUTER_PORT = 8100
ROUTER_SCRIPT = "continuum_exp/router.py"
OUTPUT_BASE_DIR = "continuum"
WORKSPACE_DIR = os.getcwd()

# 等待时间配置（秒）
VLLM_STARTUP_WAIT = 60  # 等待vllm服务启动
ROUTER_STARTUP_WAIT = 10  # 等待router启动
# ==================================================


class ProcessManager:
    """进程管理器"""

    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.log_files: List = []

    def start_vllm_server(self, gpu_devices: str, port: int,
                         scheduling_policy: str, name: str) -> subprocess.Popen:
        """启动vllm服务器"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices

        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--scheduling-policy", scheduling_policy,
            "--port", str(port),
            "--tensor-parallel-size", "4"
        ]

        log_file = open(f"{name}.log", "w")
        self.log_files.append(log_file)

        logger.info(f"启动 {name}: {' '.join(cmd)}")
        logger.info(f"GPU设备: {gpu_devices}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # 创建新的进程组
        )

        self.processes.append(process)
        return process

    def start_router(self, name: str = "router") -> subprocess.Popen:
        """启动router"""
        cmd = ["python", ROUTER_SCRIPT]

        log_file = open(f"{name}.log", "w")
        self.log_files.append(log_file)

        logger.info(f"启动 router: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

        self.processes.append(process)
        return process

    def cleanup(self):
        """清理所有进程"""
        logger.info("清理所有进程...")

        for process in self.processes:
            try:
                if process.poll() is None:  # 进程仍在运行
                    # 发送SIGTERM到整个进程组
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                    # 等待最多10秒
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        # 如果进程没有终止，强制杀死
                        logger.warning(f"进程 {process.pid} 未响应SIGTERM，发送SIGKILL")
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
            except Exception as e:
                logger.error(f"清理进程 {process.pid} 时出错: {e}")

        # 关闭日志文件
        for log_file in self.log_files:
            try:
                log_file.close()
            except:
                pass

        self.processes.clear()
        self.log_files.clear()
        logger.info("所有进程已清理")


def wait_for_server(port: int, max_retries: int = 30, interval: int = 2) -> bool:
    """等待服务器启动"""
    import socket

    logger.info(f"等待端口 {port} 上的服务启动...")

    for i in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()

            if result == 0:
                logger.info(f"端口 {port} 已就绪")
                return True
        except:
            pass

        if i < max_retries - 1:
            time.sleep(interval)

    logger.error(f"端口 {port} 启动超时")
    return False


def run_experiment(scheduling_policy: str, jps: float) -> bool:
    """运行单次实验"""
    pm = ProcessManager()

    try:
        logger.info("="*70)
        logger.info(f"开始实验: scheduling_policy={scheduling_policy}, jps={jps}")
        logger.info("="*70)

        # 1. 启动第一个vllm服务器
        logger.info("步骤 1/6: 启动第一个vllm服务器 (GPU 0,1,2,3, Port 8001)")
        pm.start_vllm_server("0,1,2,3", PORT_1, scheduling_policy,
                            f"vllm_server_1_{scheduling_policy}_{jps}")
        time.sleep(VLLM_STARTUP_WAIT)
        if not wait_for_server(PORT_1):
            logger.error("第一个vllm服务器启动失败")
            return False

        # 2. 启动第二个vllm服务器
        logger.info("步骤 2/6: 启动第二个vllm服务器 (GPU 4,5,6,7, Port 8002)")
        pm.start_vllm_server("4,5,6,7", PORT_2, scheduling_policy,
                            f"vllm_server_2_{scheduling_policy}_{jps}")
        time.sleep(VLLM_STARTUP_WAIT)
        if not wait_for_server(PORT_2):
            logger.error("第二个vllm服务器启动失败")
            return False

        # 3. 启动router
        logger.info("步骤 3/6: 启动router")
        pm.start_router(f"router_{scheduling_policy}_{jps}")
        time.sleep(ROUTER_STARTUP_WAIT)
        if not wait_for_server(ROUTER_PORT):
            logger.error("Router启动失败")
            return False

        # 4. 清理旧的swebench_router目录
        logger.info("步骤 4/6: 清理旧的swebench_router目录")
        subprocess.run(["rm", "-rf", "./swebench_router"], check=False)

        # 5. 运行mini-extra swebench
        logger.info("步骤 5/6: 运行mini-extra swebench")
        output_dir = f"./swebench_router_jps{jps}_scheduling{scheduling_policy}"

        cmd = [
            "mini-extra", "swebench",
            "--model-class", "vllm",
            "--model", MODEL_NAME,
            "--port", str(ROUTER_PORT),
            "--subset", "verified",
            "--split", "test",
            "--use-jps",
            "--jps", str(jps),
            "--output", output_dir
        ]

        logger.info(f"命令: {' '.join(cmd)}")

        with open(f"swebench_{scheduling_policy}_{jps}.log", "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        if result.returncode != 0:
            logger.error(f"mini-extra swebench 执行失败，返回码: {result.returncode}")
            return False

        logger.info("mini-extra swebench 执行成功")

        # 6. 运行分析脚本
        logger.info("步骤 6/6: 运行分析脚本")

        # 确保输出目录存在
        result_dir = Path(OUTPUT_BASE_DIR) / f"swebench_router_jps{jps}_scheduling{scheduling_policy}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "result.txt"

        analyze_cmd = [
            "python",
            "continuum_exp/analyze_client.py",
            f"{output_dir}/"
        ]

        logger.info(f"分析命令: {' '.join(analyze_cmd)}")

        with open(result_file, "w") as output_file:
            result = subprocess.run(
                analyze_cmd,
                stdout=output_file,
                stderr=subprocess.STDOUT
            )

        if result.returncode != 0:
            logger.warning(f"分析脚本执行返回码: {result.returncode}")

        logger.info(f"结果已保存到: {result_file}")
        logger.info(f"实验完成: scheduling_policy={scheduling_policy}, jps={jps}")

        return True

    except Exception as e:
        logger.error(f"实验过程中出错: {e}", exc_info=True)
        return False

    finally:
        # 清理所有进程
        pm.cleanup()

        # 额外等待，确保端口释放
        logger.info("等待端口释放...")
        time.sleep(10)


def main():
    """主函数"""
    logger.info("="*70)
    logger.info("实验脚本启动")
    logger.info("="*70)
    logger.info(f"工作目录: {WORKSPACE_DIR}")
    logger.info(f"Scheduling policies: {SCHEDULING_POLICIES}")
    logger.info(f"JPS values: {JPS_VALUES}")
    logger.info(f"总共 {len(SCHEDULING_POLICIES) * len(JPS_VALUES)} 个实验组合")
    logger.info("="*70)

    # 创建输出目录
    Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    # 遍历所有参数组合
    for i, scheduling_policy in enumerate(SCHEDULING_POLICIES):
        for j, jps in enumerate(JPS_VALUES):
            experiment_num = i * len(JPS_VALUES) + j + 1
            total_experiments = len(SCHEDULING_POLICIES) * len(JPS_VALUES)

            logger.info(f"\n{'='*70}")
            logger.info(f"实验进度: {experiment_num}/{total_experiments}")
            logger.info(f"{'='*70}\n")

            success = run_experiment(scheduling_policy, jps)

            if success:
                success_count += 1
                logger.info(f"✓ 实验成功: scheduling_policy={scheduling_policy}, jps={jps}")
            else:
                fail_count += 1
                logger.error(f"✗ 实验失败: scheduling_policy={scheduling_policy}, jps={jps}")

            # 实验之间的间隔
            if experiment_num < total_experiments:
                logger.info("等待15秒后开始下一个实验...")
                time.sleep(15)

    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("所有实验完成!")
    logger.info(f"成功: {success_count}, 失败: {fail_count}")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n收到中断信号，正在退出...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        sys.exit(1)