#!/usr/bin/env python3
"""
    System monitor node for Aion R6. Periodically samples CPU, GPU, memory and
    process usage on the Jetson and publishes all of it as a single
    diagnostic_msgs/DiagnosticArray (one DiagnosticStatus per monitor).
"""

from pathlib import Path
import socket
import psutil
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

# Jetson exposes GPU load (in per-mille) and frequency through sysfs
GPU_LOAD_PATH = Path("/sys/devices/platform/gpu.0/load")
GPU_FREQ_PATH = Path("/sys/class/devfreq/17000000.gpu/cur_freq")

OK = DiagnosticStatus.OK
WARN = DiagnosticStatus.WARN
ERROR = DiagnosticStatus.ERROR
LEVEL_MSG = {OK: "OK", WARN: "high usage", ERROR: "very high usage"}


def read_sysfs(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def kv(key, value):
    return KeyValue(key=key, value=str(value))


class SystemMonitorNode(Node):
    def __init__(self):
        super().__init__('system_monitor')

        self.declare_parameter('topic', '/system_diagnostics')
        self.declare_parameter('rate', 1.0)  # Hz
        self.declare_parameter('num_top_processes', 5)
        self.declare_parameter('usage_warn', 90.0)  # percent
        self.declare_parameter('usage_error', 95.0)  # percent
        self.declare_parameter('temp_warn', 85.0)  # deg C
        self.declare_parameter('temp_error', 95.0)  # deg C

        topic = self.get_parameter('topic').value
        rate = self.get_parameter('rate').value
        self.num_top = self.get_parameter('num_top_processes').value
        self.usage_warn = self.get_parameter('usage_warn').value
        self.usage_error = self.get_parameter('usage_error').value
        self.temp_warn = self.get_parameter('temp_warn').value
        self.temp_error = self.get_parameter('temp_error').value

        self.hardware_id = socket.gethostname()

        # cpu_percent() measures since the previous call, so prime it here
        psutil.cpu_percent(percpu=True)
        for proc in psutil.process_iter():
            try:
                proc.cpu_percent()
            except psutil.Error:
                pass

        self.publisher = self.create_publisher(DiagnosticArray, topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

        self.get_logger().info(f"Publishing system diagnostics on {topic} at {rate} Hz")

    def level_from(self, value, warn, error):
        if value >= error:
            return ERROR
        if value >= warn:
            return WARN
        return OK

    def make_status(self, name, level, values):
        return DiagnosticStatus(
            level=level,
            name=f"system_monitor: {name}",
            message=LEVEL_MSG[level],
            hardware_id=self.hardware_id,
            values=values,
        )

    def temperatures(self):
        try:
            return psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return {}

    def cpu_status(self, temps):
        per_core = psutil.cpu_percent(percpu=True)
        total = sum(per_core) / len(per_core)
        load1, load5, load15 = psutil.getloadavg()
        freq = psutil.cpu_freq()

        values = [kv("usage_total[%]", f"{total:.1f}")]
        values += [kv(f"core{i}_usage[%]", f"{p:.1f}") for i, p in enumerate(per_core)]
        values += [
            kv("load_avg_1min", f"{load1:.2f}"),
            kv("load_avg_5min", f"{load5:.2f}"),
            kv("load_avg_15min", f"{load15:.2f}"),
        ]
        if freq is not None:
            values.append(kv("frequency[MHz]", f"{freq.current:.0f}"))

        level = self.level_from(total, self.usage_warn, self.usage_error)
        cpu_temp = temps.get("cpu-thermal")
        if cpu_temp:
            values.append(kv("temperature[C]", f"{cpu_temp[0].current:.1f}"))
            level = max(level, self.level_from(cpu_temp[0].current, self.temp_warn, self.temp_error))

        return self.make_status("CPU", level, values)

    def gpu_status(self, temps):
        load = read_sysfs(GPU_LOAD_PATH)
        freq = read_sysfs(GPU_FREQ_PATH)
        if load is None:
            return DiagnosticStatus(
                level=DiagnosticStatus.STALE,
                name="system_monitor: GPU",
                message=f"cannot read {GPU_LOAD_PATH}",
                hardware_id=self.hardware_id,
            )

        usage = load / 10.0
        values = [kv("usage[%]", f"{usage:.1f}")]
        if freq is not None:
            values.append(kv("frequency[MHz]", f"{freq / 1e6:.0f}"))

        level = self.level_from(usage, self.usage_warn, self.usage_error)
        gpu_temp = temps.get("gpu-thermal")
        if gpu_temp:
            values.append(kv("temperature[C]", f"{gpu_temp[0].current:.1f}"))
            level = max(level, self.level_from(gpu_temp[0].current, self.temp_warn, self.temp_error))

        return self.make_status("GPU", level, values)

    def memory_status(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        mb = 1024 * 1024
        # Jetson GPU shares system RAM, so this includes GPU allocations
        values = [
            kv("usage[%]", f"{mem.percent:.1f}"),
            kv("total[MB]", mem.total // mb),
            kv("used[MB]", mem.used // mb),
            kv("available[MB]", mem.available // mb),
            kv("swap_usage[%]", f"{swap.percent:.1f}"),
            kv("swap_used[MB]", swap.used // mb),
            kv("swap_total[MB]", swap.total // mb),
        ]
        level = self.level_from(mem.percent, self.usage_warn, self.usage_error)
        return self.make_status("Memory", level, values)

    def process_status(self):
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'num_threads']):
            try:
                info = proc.info
                info['cpu_percent'] = proc.cpu_percent()
                procs.append(info)
            except psutil.Error:
                pass

        values = [kv("num_processes", len(procs))]
        top_cpu = sorted(procs, key=lambda p: p['cpu_percent'] or 0.0, reverse=True)
        for i, p in enumerate(top_cpu[:self.num_top]):
            values.append(kv(
                f"top_cpu_{i + 1}",
                f"{p['name']} (pid {p['pid']}): cpu {p['cpu_percent']:.1f}%, "
                f"mem {p['memory_percent'] or 0.0:.1f}%, threads {p['num_threads']}",
            ))
        top_mem = sorted(procs, key=lambda p: p['memory_percent'] or 0.0, reverse=True)
        for i, p in enumerate(top_mem[:self.num_top]):
            values.append(kv(
                f"top_mem_{i + 1}",
                f"{p['name']} (pid {p['pid']}): mem {p['memory_percent'] or 0.0:.1f}%, "
                f"cpu {p['cpu_percent']:.1f}%, threads {p['num_threads']}",
            ))

        return self.make_status("Process", OK, values)

    def timer_callback(self):
        # Timer can still fire once after Ctrl-C has shut the context down
        if not self.context.ok():
            return

        temps = self.temperatures()

        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [
            self.cpu_status(temps),
            self.gpu_status(temps),
            self.memory_status(),
            self.process_status(),
        ]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
