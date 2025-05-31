'''
Author: FoeverTree 11965818+cromwell_mao@user.noreply.gitee.com
Date: 2025-05-13 11:34:34
LastEditors: FoeverTree 11965818+cromwell_mao@user.noreply.gitee.com
LastEditTime: 2025-05-31 11:49:45
FilePath: \GPLab\src\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import asyncio
import sys
import os

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from simulation.scheduler import SimulatorScheduler

def main():
    scheduler = SimulatorScheduler(config_path='../config/config.yaml')
    try:
        asyncio.run(scheduler.run_simulation())
        # After simulation, evaluate performance (optional)
        scheduler.evaluate_simulation_performance()
    except Exception as e:
        # Use scheduler's logger if available, otherwise print
        try:
            logger = scheduler.logger
            logger.critical(f"Simulation failed with unhandled exception: {e}", exc_info=True)
        except AttributeError:
            print(f"Critical Error: Simulation failed. {e}")
        # Ensure DB connection is closed even on failure
        try:
            if scheduler.db and scheduler.db.conn:
                scheduler.db.close()
        except AttributeError:
            pass  # db might not be initialized

if __name__ == "__main__":
    main()