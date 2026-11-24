"""后续优化入口；不带参数只打印计划，不启动训练。"""
import sys
sys.dont_write_bytecode = True

if __name__ == '__main__':
    from optimization.runner import main
    main()
