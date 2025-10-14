#!/usr/bin/env python3
"""
Aura Render Management Script

Provides convenient management commands for the Aura Render application.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text"""
    print(f"{color}{text}{Colors.END}")


def print_banner():
    """Print management banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎬 AURA RENDER MANAGER 🎬                   ║
    ║              Comprehensive Application Management              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.CYAN + Colors.BOLD)


def run_command(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    if cwd is None:
        cwd = PROJECT_ROOT
    
    print_colored(f"🔧 Running: {' '.join(cmd)}", Colors.BLUE)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result
    except Exception as e:
        print_colored(f"❌ Command failed: {e}", Colors.RED)
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


class AuraRenderManager:
    """Main management class"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.pid_file = self.project_root / "aura_render.pid"
    
    def start(self, args):
        """Start the Aura Render server"""
        print_colored("🚀 Starting Aura Render server...", Colors.GREEN + Colors.BOLD)
        
        # Check if already running
        if self.is_running():
            print_colored("⚠️  Server is already running!", Colors.YELLOW)
            return 1
        
        # Choose startup script
        if args.enhanced:
            startup_script = "startup.py"
            print_colored("🔍 Using enhanced startup with diagnostics", Colors.BLUE)
        else:
            startup_script = "start.py"
            print_colored("⚡ Using standard startup", Colors.BLUE)
        
        # Start the server
        if args.daemon:
            # Start as daemon
            print_colored("🌙 Starting in daemon mode...", Colors.BLUE)
            process = subprocess.Popen([
                sys.executable, startup_script
            ], cwd=self.project_root)
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            print_colored(f"✅ Server started as daemon (PID: {process.pid})", Colors.GREEN)
            return 0
        else:
            # Start in foreground
            result = run_command([sys.executable, startup_script])
            return result.returncode
    
    def stop(self, args):
        """Stop the Aura Render server"""
        print_colored("🛑 Stopping Aura Render server...", Colors.YELLOW + Colors.BOLD)
        
        if not self.is_running():
            print_colored("ℹ️  Server is not running", Colors.BLUE)
            return 0
        
        # Get PID
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Send termination signal
            if args.force:
                print_colored("💥 Force stopping server...", Colors.RED)
                os.kill(pid, signal.SIGKILL)
            else:
                print_colored("⏹️  Gracefully stopping server...", Colors.YELLOW)
                os.kill(pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                for i in range(10):
                    if not self.is_running():
                        break
                    time.sleep(1)
                else:
                    print_colored("⚠️  Server didn't stop gracefully, force killing...", Colors.YELLOW)
                    os.kill(pid, signal.SIGKILL)
            
            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            print_colored("✅ Server stopped successfully", Colors.GREEN)
            return 0
            
        except FileNotFoundError:
            print_colored("❌ PID file not found", Colors.RED)
            return 1
        except ProcessLookupError:
            print_colored("ℹ️  Process already terminated", Colors.BLUE)
            if self.pid_file.exists():
                self.pid_file.unlink()
            return 0
        except Exception as e:
            print_colored(f"❌ Error stopping server: {e}", Colors.RED)
            return 1
    
    def restart(self, args):
        """Restart the Aura Render server"""
        print_colored("🔄 Restarting Aura Render server...", Colors.PURPLE + Colors.BOLD)
        
        # Stop first
        stop_result = self.stop(args)
        if stop_result != 0:
            print_colored("❌ Failed to stop server", Colors.RED)
            return stop_result
        
        # Wait a moment
        time.sleep(2)
        
        # Start again
        return self.start(args)
    
    def status(self, args):
        """Check server status"""
        print_colored("📊 Checking Aura Render status...", Colors.BLUE + Colors.BOLD)
        
        if self.is_running():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                print_colored(f"✅ Server is running (PID: {pid})", Colors.GREEN)
                
                # Try to get more info
                try:
                    import psutil
                    process = psutil.Process(pid)
                    print_colored(f"📈 CPU Usage: {process.cpu_percent()}%", Colors.BLUE)
                    print_colored(f"🧠 Memory Usage: {process.memory_info().rss / 1024 / 1024:.1f} MB", Colors.BLUE)
                    print_colored(f"⏱️  Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process.create_time()))}", Colors.BLUE)
                except ImportError:
                    print_colored("ℹ️  Install psutil for detailed process information", Colors.YELLOW)
                except Exception as e:
                    print_colored(f"⚠️  Cannot get process details: {e}", Colors.YELLOW)
                    
                return 0
            except Exception as e:
                print_colored(f"⚠️  Server may be running but cannot read PID: {e}", Colors.YELLOW)
                return 1
        else:
            print_colored("🔴 Server is not running", Colors.RED)
            return 1
    
    def health(self, args):
        """Run health checks"""
        print_colored("🏥 Running health checks...", Colors.BLUE + Colors.BOLD)
        
        health_script = self.project_root / "scripts" / "health_check.py"
        if not health_script.exists():
            print_colored("❌ Health check script not found", Colors.RED)
            return 1
        
        result = run_command([sys.executable, str(health_script)])
        return result.returncode
    
    def logs(self, args):
        """View logs"""
        print_colored("📄 Viewing logs...", Colors.BLUE + Colors.BOLD)
        
        log_dir = self.project_root / "logs"
        if not log_dir.exists():
            print_colored("❌ Log directory not found", Colors.RED)
            return 1
        
        # Choose log file
        if args.error:
            log_file = log_dir / "errors.log"
        else:
            log_file = log_dir / "aura_render.log"
        
        if not log_file.exists():
            print_colored(f"❌ Log file not found: {log_file}", Colors.RED)
            return 1
        
        # Display logs
        if args.follow:
            # Follow mode (like tail -f)
            try:
                result = run_command(["tail", "-f", str(log_file)])
                return result.returncode
            except KeyboardInterrupt:
                print_colored("\n👋 Log following stopped", Colors.BLUE)
                return 0
        else:
            # Show last N lines
            lines = args.lines or 50
            result = run_command(["tail", f"-{lines}", str(log_file)])
            return result.returncode
    
    def test(self, args):
        """Run tests"""
        print_colored("🧪 Running tests...", Colors.BLUE + Colors.BOLD)
        
        # Check if pytest is available
        try:
            result = run_command([sys.executable, "-m", "pytest", "--version"], capture_output=True)
            if result.returncode != 0:
                print_colored("❌ pytest not found. Install it with: pip install pytest", Colors.RED)
                return 1
        except Exception:
            print_colored("❌ pytest not found. Install it with: pip install pytest", Colors.RED)
            return 1
        
        # Run tests
        test_args = [sys.executable, "-m", "pytest"]
        
        if args.verbose:
            test_args.append("-v")
        
        if args.coverage:
            test_args.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
        
        if args.pattern:
            test_args.extend(["-k", args.pattern])
        
        # Add test directory or files
        if args.test_files:
            test_args.extend(args.test_files)
        else:
            # Default test patterns
            test_files = [
                "test_*.py",
                "*_test.py",
                "tests/"
            ]
            
            for pattern in test_files:
                test_path = self.project_root / pattern
                if test_path.exists():
                    test_args.append(str(test_path))
        
        result = run_command(test_args)
        return result.returncode
    
    def clean(self, args):
        """Clean up temporary files"""
        print_colored("🧹 Cleaning up...", Colors.BLUE + Colors.BOLD)
        
        cleanup_patterns = [
            "*.pyc",
            "__pycache__",
            "*.pyo",
            "*.pyd",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "*.log",
            "*.tmp"
        ]
        
        cleaned_count = 0
        
        for pattern in cleanup_patterns:
            if pattern.startswith("*."):
                # File patterns
                files = list(self.project_root.rglob(pattern))
                for file in files:
                    try:
                        file.unlink()
                        cleaned_count += 1
                        if args.verbose:
                            print_colored(f"🗑️  Removed: {file.relative_to(self.project_root)}", Colors.YELLOW)
                    except Exception as e:
                        if args.verbose:
                            print_colored(f"⚠️  Cannot remove {file}: {e}", Colors.YELLOW)
            else:
                # Directory patterns
                dirs = list(self.project_root.rglob(pattern))
                for dir_path in dirs:
                    if dir_path.is_dir():
                        try:
                            import shutil
                            shutil.rmtree(dir_path)
                            cleaned_count += 1
                            if args.verbose:
                                print_colored(f"🗑️  Removed directory: {dir_path.relative_to(self.project_root)}", Colors.YELLOW)
                        except Exception as e:
                            if args.verbose:
                                print_colored(f"⚠️  Cannot remove {dir_path}: {e}", Colors.YELLOW)
        
        print_colored(f"✅ Cleaned {cleaned_count} items", Colors.GREEN)
        return 0
    
    def is_running(self) -> bool:
        """Check if server is running"""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            os.kill(pid, 0)  # This doesn't actually kill, just checks if process exists
            return True
        except (FileNotFoundError, ProcessLookupError, ValueError):
            # Process doesn't exist, clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False


def main():
    """Main management function"""
    print_banner()
    
    parser = argparse.ArgumentParser(description="Aura Render Management Script")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the server')
    start_parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    start_parser.add_argument('--enhanced', action='store_true', help='Use enhanced startup with diagnostics')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop the server')
    stop_parser.add_argument('--force', action='store_true', help='Force stop (SIGKILL)')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart the server')
    restart_parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    restart_parser.add_argument('--enhanced', action='store_true', help='Use enhanced startup')
    restart_parser.add_argument('--force', action='store_true', help='Force stop before restart')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check server status')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Run health checks')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View logs')
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow log output')
    logs_parser.add_argument('--error', action='store_true', help='Show error logs')
    logs_parser.add_argument('--lines', '-n', type=int, help='Number of lines to show')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    test_parser.add_argument('--pattern', '-k', help='Test pattern to match')
    test_parser.add_argument('test_files', nargs='*', help='Specific test files to run')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean temporary files')
    clean_parser.add_argument('--verbose', '-v', action='store_true', help='Show cleaned files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = AuraRenderManager()
    
    # Map commands to methods
    command_map = {
        'start': manager.start,
        'stop': manager.stop,
        'restart': manager.restart,
        'status': manager.status,
        'health': manager.health,
        'logs': manager.logs,
        'test': manager.test,
        'clean': manager.clean
    }
    
    if args.command in command_map:
        try:
            return command_map[args.command](args)
        except KeyboardInterrupt:
            print_colored("\n⏹️  Command interrupted by user", Colors.YELLOW)
            return 130
        except Exception as e:
            print_colored(f"❌ Command failed: {e}", Colors.RED)
            return 1
    else:
        print_colored(f"❌ Unknown command: {args.command}", Colors.RED)
        return 1


if __name__ == "__main__":
    sys.exit(main())