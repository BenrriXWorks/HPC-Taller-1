#!/usr/bin/env python3
"""
Comprehensive benchmark for SKA_SORT vs VERGESORT Python Bindings
"""

import ska_sort_py
import vergesort_py
import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

console = Console()

def time_function(func, *args, iterations=10):
    """Time a function over multiple iterations"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)

def create_test_data(size, dtype):
    """Create test data for benchmarking"""
    if dtype in [np.int8, np.int16, np.int32, np.int64]:
        info = np.iinfo(dtype)
        return np.random.randint(info.min//2, info.max//2, size=size, dtype=dtype)
    elif dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        info = np.iinfo(dtype)
        return np.random.randint(0, info.max//2, size=size, dtype=dtype)
    elif dtype == np.bool_:
        return np.random.choice([True, False], size=size)
    elif dtype in [np.float32, np.float64]:
        return np.random.uniform(-1000.0, 1000.0, size=size).astype(dtype)
    elif isinstance(dtype, str) and dtype.startswith('S'):
        max_len = int(dtype[1:]) if len(dtype) > 1 else 10
        words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape']
        strings = []
        for _ in range(size):
            if np.random.random() < 0.7:
                word = np.random.choice(words)
                strings.append(word[:max_len-1] if len(word) >= max_len else word)
            else:
                length = np.random.randint(1, min(max_len, 8))
                random_str = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))
                strings.append(random_str)
        return np.array(strings, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def format_time(t):
    """Format time with colors"""
    if t == float('inf'): return "[red]ERROR[/red]"
    elif t >= 1.0: return f"[red]{t:.3f}s[/red]"
    elif t >= 0.001: return f"[yellow]{t*1000:.2f}ms[/yellow]"
    else: return f"[green]{t*1000000:.1f}μs[/green]"

def format_speedup(s):
    """Format speedup with colors"""
    if s <= 0 or s == float('inf'): return "[dim]N/A[/dim]"
    elif s >= 2.0: return f"[bold green]{s:.2f}x[/bold green]"
    elif s >= 1.1: return f"[green]{s:.2f}x[/green]"
    elif s >= 0.9: return f"[yellow]{s:.2f}x[/yellow]"
    else: return f"[red]{s:.2f}x[/red]"

def benchmark_algorithms(arr):
    """Benchmark all algorithms"""
    algorithms = {
        'ska': lambda x: ska_sort_py.sort(x),
        'verge': lambda x: vergesort_py.sort(x), 
        'numpy': lambda x: x.sort(),
        'python': lambda x: x.__setitem__(slice(None), sorted(x))
    }
    
    results = {}
    
    for name, sort_func in algorithms.items():
        arr_copy = arr.copy()
        try:
            if name == 'python':
                # Convert to list for Python sorting, then back to array
                arr_list = arr_copy.tolist()
                results[name] = time_function(lambda: arr_list.sort())
            else:
                results[name] = time_function(sort_func, arr_copy)
        except Exception:
            results[name] = float('inf')
    
    return results

def test_correctness():
    """Quick correctness test"""
    console.print(Panel("🧪 Running correctness tests...", border_style="green"))
    # Basic manual test cases
    basic_cases = [
        (np.array([5, 2, 8, 1, 9], dtype=np.int32), "Basic int32"),
        (np.array([3.14, -2.71, 0.0, 1.41], dtype=np.float32), "Basic float32"),
        (np.array([True, False, True, False], dtype=np.bool_), "Basic bool"),
        (np.array(['zebra', 'apple', 'mango', 'banana'], dtype='S10'), "Basic strings"),
        (np.array([], dtype=np.int32), "Empty array"),
        (np.array([42], dtype=np.int32), "Single element"),
    ]
    # Random test cases (100 elements each type)
    random_cases = [
        (create_test_data(100, np.int32), "Random int32 (100)"),
        (create_test_data(100, np.float64), "Random float64 (100)"),
        (create_test_data(100, np.bool_), "Random bool (100)"),
        (create_test_data(100, "S10"), "Random strings (100)"),
    ]
    all_cases = basic_cases + random_cases
    table = Table(title="Correctness Results", header_style="bold green")
    table.add_column("Test", style="cyan", width=25)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Details", width=40)
    all_passed = True
    algorithms = [(ska_sort_py.sort, "SKA_SORT"), (vergesort_py.sort, "VERGESORT")]
    
    for arr, desc in all_cases:
        passed = []
        failed = []
        errored = []
        
        if len(arr) == 0:
            table.add_row(desc, "[green]✅ PASS[/green]", "Empty array - trivially sorted")
            continue
        
        # Get expected result using Python sort
        if arr.dtype.kind == 'S':
            arr_list = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in arr]
            expected = np.array(sorted(arr_list), dtype=arr.dtype)
        else:
            expected = np.array(sorted(arr.tolist()), dtype=arr.dtype)
        
        for algo, name in algorithms:
            try:
                arr_test = arr.copy()
                algo(arr_test)
                if np.array_equal(arr_test, expected):
                    passed.append(name)
                else:
                    failed.append(name)
            except Exception as e:
                errored.append(f"{name}: {str(e)[:30]}")
        
        if passed and not failed and not errored:
            table.add_row(desc, "[green]✅ PASS[/green]", f"Passed: {passed}")
        elif passed and (failed or errored):
            details = f"Passed: {passed}"
            if failed:
                details += f" | Failed: {failed}"
            if errored:
                details += f" | Error: {errored}"
            table.add_row(desc, "[yellow]⚠️ PARTIAL[/yellow]", details)
            all_passed = False
        else:
            details = ""
            if failed:
                details += f"Failed: {failed} "
            if errored:
                details += f"Error: {errored}"
            table.add_row(desc, "[red]❌ FAIL[/red]", details)
            all_passed = False
    
    console.print(table)
    if all_passed:
        print("All correctness tests passed!\n")
    return all_passed

def create_benchmark_table(results_data):
    """Create benchmark results table"""
    table = Table(title="🚀 SKA_SORT vs VERGESORT vs NumPy vs Python Performance", show_header=True, header_style="bold magenta")
    
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Size", justify="right", style="blue", width=10)
    table.add_column("SKA Sort", justify="right", width=12)
    table.add_column("Vergesort", justify="right", width=12)
    table.add_column("NumPy", justify="right", width=12)
    table.add_column("Python", justify="right", width=12)
    table.add_column("SKA vs NumPy", justify="right", width=12)
    table.add_column("VG vs NumPy", justify="right", width=12)
    
    for result in results_data:
        ska_speedup = result['numpy'] / result['ska'] if result['ska'] > 0 else 0
        verge_speedup = result['numpy'] / result['verge'] if result['verge'] > 0 else 0
        
        table.add_row(
            result['type'],
            f"{result['size']:,}",
            format_time(result['ska']),
            format_time(result['verge']),
            format_time(result['numpy']),
            format_time(result.get('python', float('inf'))),
            format_speedup(ska_speedup),
            format_speedup(verge_speedup)
        )
    
    return table

def apply_dracula_theme():
    """Apply Dracula dark theme to matplotlib"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#282a36',
        'axes.facecolor': '#282a36',
        'axes.edgecolor': '#6272a4',
        'axes.labelcolor': '#f8f8f2',
        'text.color': '#f8f8f2',
        'xtick.color': '#f8f8f2',
        'ytick.color': '#f8f8f2',
        'grid.color': '#44475a',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.edgecolor': '#282a36',
        'savefig.facecolor': '#282a36',
        'savefig.edgecolor': '#282a36'
    })

def create_performance_plots(results_data):
    """Create performance visualization plots with Dracula theme"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        console.print("[yellow]Matplotlib or pandas not available. Skipping plot generation.[/yellow]")
        return
    
    if not results_data:
        console.print("[yellow]No data available for plotting.[/yellow]")
        return
    
    apply_dracula_theme()
    
    # Convert data to pandas DataFrame for easier manipulation
    df = pd.DataFrame(results_data)
    
    # Define colors for each algorithm
    colors = {
        'ska': '#50fa7b',     # Dracula green
        'verge': '#ff79c6',   # Dracula pink  
        'numpy': '#8be9fd',   # Dracula cyan
        'python': '#bd93f9'   # Dracula purple
    }
    
    # Get unique data types for subplots
    unique_types = df['type'].unique()
    n_types = len(unique_types)
    
    # Create subplots - adapt layout based on number of types
    if n_types <= 2:
        fig, axes = plt.subplots(1, n_types, figsize=(7*n_types, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ensure axes is always a list for consistency
    if n_types == 1:
        axes = [axes]
    elif n_types <= 4:
        axes = axes.flatten()
    
    fig.suptitle('SKA_SORT vs VERGESORT vs NumPy Performance Benchmark', 
                 fontsize=16, fontweight='bold', color='#f8f8f2')
    
    # Plot performance for each data type
    for i, data_type in enumerate(unique_types):
        if i >= len(axes):
            break
            
        ax = axes[i]
        type_data = df[df['type'] == data_type].sort_values('size')
        
        if type_data.empty:
            ax.set_visible(False)
            continue
        
        sizes = type_data['size']
        
        # Plot each algorithm
        algorithms = ['ska', 'verge', 'numpy', 'python']
        markers = ['o', 's', '^', 'x']
        
        for alg, marker in zip(algorithms, markers):
            if alg in type_data.columns:
                times_ms = type_data[alg] * 1000  # Convert to milliseconds
                
                # Filter out invalid times (inf or very large values)
                valid_mask = (times_ms != float('inf')) & (times_ms < 1e6)
                if valid_mask.any():
                    valid_sizes = sizes[valid_mask]
                    valid_times = times_ms[valid_mask]
                    
                    ax.plot(valid_sizes, valid_times, marker=marker, linestyle='-', 
                           color=colors[alg], label=alg.upper(), linewidth=2, markersize=8)
        
        # Set logarithmic scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Customize appearance
        ax.set_title(f'{data_type.upper()} Performance', fontweight='bold', color='#f8f8f2')
        ax.set_xlabel('Array Size', color='#f8f8f2')
        ax.set_ylabel('Time (ms)', color='#f8f8f2')
        ax.grid(True, alpha=0.3, color='#6272a4')
        ax.legend(facecolor='#44475a', edgecolor='#6272a4', loc='upper left')
        
        # Custom x-axis labels
        if not type_data.empty:
            unique_sizes = sorted(type_data['size'].unique())
            ax.set_xticks(unique_sizes)
            labels = []
            for s in unique_sizes:
                if s >= 1000000:
                    labels.append(f'{s//1000000}M')
                elif s >= 1000:
                    labels.append(f'{s//1000}K')
                else:
                    labels.append(str(s))
            ax.set_xticklabels(labels)
    
    # Hide unused subplots
    for i in range(n_types, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_filename = 'benchmark_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='#282a36')
    console.print(f"[green]📊 Performance plots saved as '{plot_filename}'[/green]")
    
    # Create speedup comparison plot
    fig2, ax = plt.subplots(figsize=(14, 8))
    fig2.patch.set_facecolor('#282a36')
    ax.set_facecolor('#282a36')
    
    # Filter out invalid results
    valid_results = [r for r in results_data if r['ska'] != float('inf') and r['verge'] != float('inf') and r['numpy'] != float('inf')]
    
    if not valid_results:
        console.print("[yellow]No valid results for speedup comparison.[/yellow]")
        return
    
    x_pos = np.arange(len(valid_results))
    labels = []
    ska_speedups = []
    verge_speedups = []
    
    for r in valid_results:
        # Create label
        if r['size'] >= 1000000:
            size_label = f"{r['size']//1000000}M"
        elif r['size'] >= 1000:
            size_label = f"{r['size']//1000}K"
        else:
            size_label = str(r['size'])
        labels.append(f"{r['type']}\n{size_label}")
        
        # Calculate speedups
        ska_speedups.append(r['numpy'] / r['ska'] if r['ska'] > 0 else 0)
        verge_speedups.append(r['numpy'] / r['verge'] if r['verge'] > 0 else 0)
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, ska_speedups, width, label='SKA_SORT vs NumPy', 
                   color=colors['ska'], alpha=0.9, edgecolor='#282a36')
    bars2 = ax.bar(x_pos + width/2, verge_speedups, width, label='VERGESORT vs NumPy', 
                   color=colors['verge'], alpha=0.9, edgecolor='#282a36')
    
    ax.set_ylabel('Speedup vs NumPy', color='#f8f8f2')
    ax.set_title('Speedup Comparison: SKA_SORT & VERGESORT vs NumPy', 
                fontweight='bold', color='#f8f8f2')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', color='#f8f8f2')
    ax.legend(facecolor='#44475a', edgecolor='#6272a4')
    ax.grid(True, alpha=0.3, axis='y', color='#6272a4')
    ax.axhline(y=1, color='#ffb86c', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    # Add value annotations on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                       fontsize=9, color='#f8f8f2', weight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                       fontsize=9, color='#f8f8f2', weight='bold')
    
    plt.tight_layout()
    speedup_filename = 'speedup_comparison.png'
    plt.savefig(speedup_filename, dpi=300, bbox_inches='tight', facecolor='#282a36')
    console.print(f"[green]🚀 Speedup plots saved as '{speedup_filename}'[/green]")
    
    plt.show()

def run_benchmark(sizes=None):
    """Run benchmark with configurable sizes"""
    if sizes is None:
        sizes = [100, 1000, 10000, 100000, 1000000]
    console.print(Panel(f"SKA_SORT v{ska_sort_py.__version__} vs VERGESORT v{vergesort_py.__version__}", 
                       title="🔧 Benchmark", border_style="blue"))
    test_configs = [
        (np.int32, "int32"),
        (np.float64, "float64"),
        (np.bool_, "bool"),
        ("S10", "string")
    ]
    results = []
    for dtype, type_name in test_configs:
        for size in sizes:
            console.print(f"[dim]Testing {type_name} with {size:,} elements...[/dim]")
            try:
                arr = create_test_data(size, dtype)
                bench_results = benchmark_algorithms(arr)
                results.append({
                    'type': type_name,
                    'size': size,
                    'ska': bench_results['ska'],
                    'verge': bench_results['verge'], 
                    'numpy': bench_results['numpy'],
                    'python': bench_results['python']
                })
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    table = create_benchmark_table(results)
    console.print(table)
    
    # Display summary statistics
    valid_ska = [r for r in results if r['ska'] != float('inf')]
    valid_verge = [r for r in results if r['verge'] != float('inf')]
    
    if valid_ska:
        avg_ska_speedup = sum(r['numpy']/r['ska'] for r in valid_ska) / len(valid_ska)
        console.print(f"\n[green]📊 Average SKA_SORT speedup vs NumPy: {avg_ska_speedup:.2f}x[/green]")
    
    if valid_verge:
        avg_verge_speedup = sum(r['numpy']/r['verge'] for r in valid_verge) / len(valid_verge)
        console.print(f"[green]📊 Average VERGESORT speedup vs NumPy: {avg_verge_speedup:.2f}x[/green]")
    
    # Generate performance plots
    create_performance_plots(results)

if __name__ == "__main__":
    console.print("[bold blue]🚀 SKA_SORT vs VERGESORT Benchmark Suite[/bold blue]\n")
    passed = test_correctness()
    if not passed:
        console.print("[yellow]⚠️  Some correctness tests failed. Review the table above.[/yellow]\n")
    
    console.print()
    console.print("[bold]Choose benchmark:[/bold]")        
    console.print("1. [green]Quick test[/green] (5K elements)")
    console.print("2. [yellow]Full benchmark[/yellow] (100-1M elements)")
    console.print("3. [red]Full benchmark + 10M[/red] (⚠️  high memory usage)")
    
    try:
        choice = input("\nChoice (1-3, default=1): ").strip() or "1"
        if choice == "1":
            console.print(Panel("🏁 Running quick test...", border_style="green"))
            quick_results = []
            for dtype, name in [(np.int32, "int32"), (np.float64, "float64"), ("S10", "string")]:
                console.print(f"[dim]Testing {name}...[/dim]")
                arr = create_test_data(5000, dtype)
                bench = benchmark_algorithms(arr)
                quick_results.append({
                    'type': name, 'size': 5000, 
                    'ska': bench['ska'], 'verge': bench['verge'], 
                    'numpy': bench['numpy'], 'python': bench['python']
                })
            table = create_benchmark_table(quick_results)
            console.print(table)
            create_performance_plots(quick_results)
        elif choice == "2":
            run_benchmark([100, 1000, 10000, 100000, 1000000])
        elif choice == "3":
            run_benchmark([100, 1000, 10000, 100000, 1000000, 10000000])
        else:
            console.print("[red]Invalid choice. Use 1, 2, or 3.[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
