import time
import sys
import os
import threading
import queue
import re

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
SYSTEM_MEMORY_SIZE = 1024  # Simulated "blocks" of RAM
TIME_SLICE = 0.5           # Scheduler time slice in seconds

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def print(text, color=ENDC):
        print(f"{color}{text}{Colors.ENDC}")

# ==========================================
# MODULE 1: MEMORY MANAGEMENT UNIT (MMU)
# ==========================================
class MemoryManager:
    def __init__(self, size):
        self.size = size
        self.memory_map = [None] * size  # None = Free, PID = Occupied
    
    def allocate(self, pid, blocks_needed):
        """First-fit allocation strategy."""
        free_counter = 0
        start_index = -1
        
        for i in range(self.size):
            if self.memory_map[i] is None:
                if free_counter == 0:
                    start_index = i
                free_counter += 1
                if free_counter == blocks_needed:
                    # Found space, allocate it
                    for j in range(start_index, start_index + blocks_needed):
                        self.memory_map[j] = pid
                    return start_index
            else:
                free_counter = 0
                start_index = -1
        return -1 # Out of memory

    def free(self, pid):
        """Free all memory associated with a PID."""
        freed = 0
        for i in range(self.size):
            if self.memory_map[i] == pid:
                self.memory_map[i] = None
                freed += 1
        return freed

    def get_status(self):
        used = sum(1 for x in self.memory_map if x is not None)
        return used, self.size

# ==========================================
# MODULE 2: VIRTUAL FILE SYSTEM (VFS)
# ==========================================
class FileNode:
    def __init__(self, name, is_dir=False, content=""):
        self.name = name
        self.is_dir = is_dir
        self.content = content
        self.children = {} # Only for directories
        self.created_at = time.strftime("%H:%M:%S")

class VirtualFileSystem:
    def __init__(self):
        self.root = FileNode("/", is_dir=True)
        # Create default system structure
        self.root.children["bin"] = FileNode("bin", is_dir=True)
        self.root.children["home"] = FileNode("home", is_dir=True)
        self.root.children["var"] = FileNode("var", is_dir=True)
        
        # Add a demo script
        demo_code = (
            "SET count 0\n"
            "LABEL start\n"
            "ADD count 1\n"
            "PRINT count\n"
            "IF count < 5 JUMP start\n"
            "PRINT Done!"
        )
        self.root.children["home"].children["demo.asm"] = FileNode("demo.asm", content=demo_code)
        
        self.current_path = ["/"]
        self.current_node = self.root

    def _get_node_from_path(self, path):
        """Helper to navigate absolute or relative paths."""
        if path == "/": return self.root
        parts = path.split("/")
        curr = self.current_node
        
        # Determine start point
        if path.startswith("/"):
            curr = self.root
            parts.pop(0) # Remove empty string from split
            
        for part in parts:
            if part == "..":
                # Only simulated simple parent traversal for this demo
                curr = self.root # Fallback to root for safety in this simple impl
            elif part in curr.children and curr.children[part].is_dir:
                curr = curr.children[part]
            elif part == "" or part == ".":
                continue
            else:
                return None
        return curr

    def ls(self):
        items = []
        for name, node in self.current_node.children.items():
            kind = "DIR " if node.is_dir else "FILE"
            items.append(f"{kind} | {name.ljust(15)} | {len(node.content)}b")
        return "\n".join(items)

    def mkdir(self, name):
        if name in self.current_node.children:
            return "Error: Exists"
        self.current_node.children[name] = FileNode(name, is_dir=True)
        return "Directory created."

    def touch(self, name, content=""):
        self.current_node.children[name] = FileNode(name, content=content)
        return f"File {name} created."

    def cd(self, path):
        target = self._get_node_from_path(path)
        if target:
            self.current_node = target
            if path == "/": self.current_path = ["/"]
            else: self.current_path.append(target.name + "/")
            return ""
        return "Error: Path not found."

    def cat(self, name):
        if name in self.current_node.children and not self.current_node.children[name].is_dir:
            return self.current_node.children[name].content
        return "Error: File not found or is directory."

# ==========================================
# MODULE 3: PROCESS MANAGEMENT & INTERPRETER
# ==========================================
class PCB:
    """Process Control Block"""
    def __init__(self, pid, name, code, parent_pid=0):
        self.pid = pid
        self.name = name
        self.state = "READY" # READY, RUNNING, WAITING, TERMINATED
        self.pc = 0 # Program Counter
        self.memory_start = -1
        self.code = code.split('\n')
        self.vars = {}
        self.labels = self._scan_labels()

    def _scan_labels(self):
        labels = {}
        for idx, line in enumerate(self.code):
            parts = line.strip().split()
            if parts and parts[0] == "LABEL":
                labels[parts[1]] = idx
        return labels

class ProcessScheduler:
    def __init__(self, memory_manager, kernel_io_callback):
        self.next_pid = 1
        self.ready_queue = queue.Queue()
        self.process_table = {}
        self.mm = memory_manager
        self.kernel_io = kernel_io_callback # Callback to print to screen
        self.active = False
        self.current_process = None

    def create_process(self, name, code):
        # 1. Allocate Memory
        mem_needed = max(1, len(code) // 100) # Simple metric
        addr = self.mm.allocate(self.next_pid, mem_needed)
        
        if addr == -1:
            return "Error: Out of Memory"

        # 2. Create PCB
        pcb = PCB(self.next_pid, name, code)
        pcb.memory_start = addr
        
        # 3. Add to structures
        self.process_table[self.next_pid] = pcb
        self.ready_queue.put(pcb)
        
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def kill_process(self, pid):
        if pid in self.process_table:
            self.process_table[pid].state = "TERMINATED"
            self.mm.free(pid)
            del self.process_table[pid]
            return True
        return False

    def list_processes(self):
        res = "PID  | STATE    | MEMORY | NAME\n"
        res += "-----|----------|--------|-----\n"
        for pid, pcb in self.process_table.items():
            res += f"{str(pid).ljust(4)} | {pcb.state.ljust(8)} | {str(pcb.memory_start).ljust(6)} | {pcb.name}\n"
        return res

    def run_cycle(self):
        """
        Execute one instruction of the current process (Round Robin simulated).
        Since Python is single threaded in this file context, we run one instruction
        then yield.
        """
        if self.ready_queue.empty():
            return

        pcb = self.ready_queue.get()
        if pcb.state == "TERMINATED": return

        self.current_process = pcb
        pcb.state = "RUNNING"
        
        # Execute Instruction
        try:
            if pcb.pc < len(pcb.code):
                line = pcb.code[pcb.pc].strip()
                self._execute_instruction(pcb, line)
                pcb.pc += 1
                pcb.state = "READY"
                self.ready_queue.put(pcb) # Back of the line
            else:
                pcb.state = "TERMINATED"
                self.kernel_io(f"\n[KERNEL] Process {pcb.pid} ({pcb.name}) finished.\n")
                self.mm.free(pcb.pid)
                del self.process_table[pcb.pid]
                self.current_process = None
        except Exception as e:
            self.kernel_io(f"\n[KERNEL] Process {pcb.pid} crashed: {e}\n")
            self.kill_process(pcb.pid)

    def _execute_instruction(self, pcb, line):
        if not line or line.startswith("#"): return

        parts = line.split()
        cmd = parts[0].upper()

        if cmd == "SET": # SET var val
            val = self._resolve_val(pcb, parts[2])
            pcb.vars[parts[1]] = val
        
        elif cmd == "ADD": # ADD var val
            val = self._resolve_val(pcb, parts[2])
            pcb.vars[parts[1]] += val

        elif cmd == "SUB":
            val = self._resolve_val(pcb, parts[2])
            pcb.vars[parts[1]] -= val

        elif cmd == "PRINT":
            msg = line[6:]
            if msg in pcb.vars:
                output = str(pcb.vars[msg])
            else:
                # Check if it's a raw string or variable
                output = msg
            self.kernel_io(f"[Process {pcb.pid}]: {output}")

        elif cmd == "JUMP":
            if parts[1] in pcb.labels:
                pcb.pc = pcb.labels[parts[1]]

        elif cmd == "IF": # IF var < 10 JUMP label
            var_val = self._resolve_val(pcb, parts[1])
            op = parts[2]
            comp_val = self._resolve_val(pcb, parts[3])
            dest = parts[5]

            res = False
            if op == "==": res = var_val == comp_val
            elif op == "<": res = var_val < comp_val
            elif op == ">": res = var_val > comp_val
            
            if res and dest in pcb.labels:
                pcb.pc = pcb.labels[dest]
        
        elif cmd == "LABEL":
            pass # Handle in scan
        
        # Add CPU delay simulation
        time.sleep(0.05) 

    def _resolve_val(self, pcb, token):
        if token in pcb.vars:
            return pcb.vars[token]
        try:
            return int(token)
        except:
            return str(token)

# ==========================================
# MODULE 4: KERNEL
# ==========================================
class Kernel:
    def __init__(self):
        self.running = True
        self.mm = MemoryManager(SYSTEM_MEMORY_SIZE)
        self.fs = VirtualFileSystem()
        self.scheduler = ProcessScheduler(self.mm, self.io_interrupt)
        self.start_time = time.time()

    def boot(self):
        self.clear_screen()
        Colors.print("========================================", Colors.HEADER)
        Colors.print("      PyMinOS v1.0 - Boot Sequence      ", Colors.HEADER)
        Colors.print("========================================", Colors.HEADER)
        time.sleep(0.5)
        print("[ OK ] Initializing Memory Management Unit...")
        time.sleep(0.3)
        print("[ OK ] Mounting Virtual File System (VFS)...")
        time.sleep(0.3)
        print("[ OK ] Starting Process Scheduler...")
        print(f"[INFO] System Memory: {SYSTEM_MEMORY_SIZE} blocks")
        print("\nWelcome User. Type 'help' for commands.")

    def shutdown(self):
        Colors.print("\nSystem shutting down...", Colors.WARNING)
        self.running = False

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def io_interrupt(self, message):
        """Handles IO output from processes safely."""
        print(message)

    def execute_shell_command(self, cmd_str):
        parts = cmd_str.strip().split()
        if not parts: return

        cmd = parts[0].lower()
        args = parts[1:]

        # --- File System Commands ---
        if cmd == "ls":
            print(self.fs.ls())
        elif cmd == "mkdir":
            if not args: print("Usage: mkdir <name>")
            else: print(self.fs.mkdir(args[0]))
        elif cmd == "touch":
            if not args: print("Usage: touch <name>")
            else: print(self.fs.touch(args[0]))
        elif cmd == "cd":
            if not args: print("Usage: cd <path>")
            else: print(self.fs.cd(args[0]))
        elif cmd == "cat":
            if not args: print("Usage: cat <file>")
            else: print(self.fs.cat(args[0]))
        elif cmd == "write":
            self.handle_write(args)

        # --- Process Commands ---
        elif cmd == "ps":
            print(self.scheduler.list_processes())
        elif cmd == "kill":
            if not args: print("Usage: kill <pid>")
            else: 
                try: 
                    self.scheduler.kill_process(int(args[0]))
                    print("Process killed.")
                except: print("Invalid PID")
        elif cmd == "exec":
            if not args: print("Usage: exec <filename>")
            else: self.launch_program(args[0])

        # --- System Utilities ---
        elif cmd == "calc":
            try:
                # Safe eval
                allowed = set("0123456789+-*/(). ")
                if set(" ".join(args)) <= allowed:
                    print(eval(" ".join(args)))
                else:
                    print("Security Error: Invalid characters")
            except Exception as e:
                print(f"Error: {e}")
        elif cmd == "mem":
            used, total = self.mm.get_status()
            bar = "#" * int((used/total)*20)
            empty = "-" * (20 - len(bar))
            print(f"RAM: [{bar}{empty}] {used}/{total} blocks")
        elif cmd == "clear":
            self.clear_screen()
        elif cmd == "help":
            print("Commands: ls, cd, mkdir, touch, cat, write, ps, kill, exec, calc, mem, exit")
        elif cmd == "exit":
            self.shutdown()
        else:
            print(f"Unknown command: {cmd}")

    def handle_write(self, args):
        if not args: 
            print("Usage: write <filename>")
            return
        filename = args[0]
        print(f"Writing to {filename}. Type 'EOF' on a new line to save.")
        lines = []
        while True:
            line = input(">> ")
            if line == "EOF": break
            lines.append(line)
        self.fs.touch(filename, "\n".join(lines))
        print("File saved.")

    def launch_program(self, filename):
        content = self.fs.cat(filename)
        if content.startswith("Error"):
            print(content)
            return
        
        pid = self.scheduler.create_process(filename, content)
        if isinstance(pid, int):
            print(f"Started Process PID: {pid}")
            # In a real OS, this runs in background. 
            # Here we loop the scheduler until queue empty for demonstration
            # or the user can run 'step' command. 
            # For UX, we will run it immediately until blocked or done.
            while not self.scheduler.ready_queue.empty():
                self.scheduler.run_cycle()
        else:
            print(pid) # Error message

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    kernel = Kernel()
    kernel.boot()

    while kernel.running:
        try:
            # Construct prompt
            path = "".join(kernel.fs.current_path)
            prompt = f"{Colors.GREEN}root@PyMinOS{Colors.ENDC}:{Colors.BLUE}{path}{Colors.ENDC}$ "
            
            cmd = input(prompt)
            kernel.execute_shell_command(cmd)
            
        except KeyboardInterrupt:
            kernel.shutdown()
            break
        except Exception as e:
            Colors.print(f"Kernel Panic: {e}", Colors.FAIL)