import sys
import numpy as np

def memory_report(my_instance):
    """
    """
    print(f"\n{"="*20} INSTANCE MEMORY REPORT {"="*20}")
    print(f"Target Class: {my_instance.__class__.__name__}")
    print(f"{"Variable Name":<30} | {"Memory Size":<15} | {"Type":<20}")
    print("-" * 73)
    
    total_bytes = 0
    report_list = []
    
    # Loop over all instance attributes
    attrs = getattr(my_instance, '__dict__', {})
    for attr_name, attr_val in attrs.items():
        size_bytes = 0
        
        # 1. NumPy array
        if isinstance(attr_val, np.ndarray):
            size_bytes = attr_val.nbytes
        
        # 2. List and tupple
        elif isinstance(attr_val, (list, tuple)):
            size_bytes = sys.getsizeof(attr_val)
            for item in attr_val:
                if isinstance(item, np.ndarray):
                    size_bytes += item.nbytes
                else:
                    size_bytes += sys.getsizeof(item)
                    
        # 3. Other objects
        else:
            size_bytes = sys.getsizeof(attr_val)
            
        total_bytes += size_bytes
        report_list.append((attr_name, size_bytes, type(attr_val).__name__))
        
    # Sort in the size-descending order
    report_list.sort(key=lambda x: x[1], reverse=True)
    
    for name, size, t_name in report_list:
        if size >= 1024**3:
            size_str = f"{size / (1024**3):.2f} GB"
        elif size >= 1024**2:
            size_str = f"{size / (1024**2):.2f} MB"
        else:
            size_str = f"{size / 1024:.2f} KB"
            
        print(f"{name:<30} | {size_str:<15} | {t_name:<20}")
        
    print("-" * 73)
    total_gb = total_bytes / (1024**3)
    print(f"💥 TOTAL ESTIMATED MEMORY: {total_gb:.2f} GB")
    print(f"{"="*67}\n")
