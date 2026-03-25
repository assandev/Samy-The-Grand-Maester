from pathlib import Path


def clear_directory(directory: Path, keep_gitkeep: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if keep_gitkeep and item.name == ".gitkeep":
            continue
        if item.is_dir():
            clear_directory(item, keep_gitkeep=False)
            item.rmdir()
        else:
            item.unlink()


def reset_runtime_data(*directories: Path) -> None:
    for directory in directories:
        clear_directory(directory)
