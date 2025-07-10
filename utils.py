import os

def replace_suffix(text, suffix, replacement):
    """
    Replace the last occurrence of a suffix in a string with a replacement string.
    
    Args:
        text (str): The original text.
        suffix (str): The suffix to replace.
        replacement (str): The string to replace the suffix with.
    
    Returns:
        str: The modified text with the suffix replaced.
    """
    if text.endswith(suffix):
        return text[:-len(suffix)] + replacement
    return text


def generate_unique_filename(base_name, extension, start_index=1, separator="_", padding=0):
    """
    Generate a unique filename by appending an incrementing index.
    
    Args:
        base_name (str): Base name for the file
        extension (str): File extension (with or without leading dot)
        start_index (int): Starting index (default: 1)
        separator (str): Separator between base name and index (default: "_")
        padding (int): Zero-padding width for index (default: 0 = no padding)
    
    Returns:
        str: Unique filename that doesn't exist
    """
    # Normalize extension to include leading dot
    if not extension.startswith("."):
        extension = "." + extension
    
    index = start_index
    while True:
        # Format index with zero-padding
        index_str = f"{index:0{padding}d}" if padding else str(index)
        filename = f"{base_name}{separator}{index_str}{extension}"
        
        # Check if file exists
        if not os.path.exists(filename):
            return filename
            
        index += 1

# Example usage
if __name__ == "__main__":
    new_file = generate_unique_filename(
        base_name="data",
        extension="txt",       # Can use ".txt" or "txt"
        start_index=1,
        separator="_",
        padding=3              # Try 0 for no padding
    )
    print("Generated filename:", new_file)