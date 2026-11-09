"""已保存实验的评测入口。

示例：python inference_plv.py --data <数据目录> --output <训练运行目录>
      --model tcn --eval-split test --device cuda
默认评测 val；test 需显式指定。模型类型须与训练时一致。
output 中须已有 run.json、state.npz 和对应模型文件。
"""
import sys
from engine_plv import main

if __name__ == "__main__":
    # 强制 eval-only：复用 engine 中的标准化、检查点加载及指标导出流程。
    main([*sys.argv[1:], "--eval-only"])
