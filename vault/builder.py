from .version import Version

def build_version(parent, task_name, new_classes=10):
    return Version(parent, task_name, new_classes)
