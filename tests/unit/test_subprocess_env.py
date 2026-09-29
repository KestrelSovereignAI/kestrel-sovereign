"""Cross-platform contract for secret-free subprocess environments."""

from kestrel_sovereign.security.subprocess_env import sanitized_subprocess_env


def test_windows_runtime_context_survives_secret_filtering():
    source = {
        "SYSTEMROOT": r"C:\\Windows",
        "WINDIR": r"C:\\Windows",
        "SYSTEMDRIVE": "C:",
        "TEMP": r"C:\\Users\\bird\\Temp",
        "TMP": r"C:\\Users\\bird\\Temp",
        "USERPROFILE": r"C:\\Users\\bird",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\\Users\\bird",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "COMSPEC": r"C:\\Windows\\System32\\cmd.exe",
        "APPDATA": r"C:\\Users\\bird\\AppData\\Roaming",
        "LOCALAPPDATA": r"C:\\Users\\bird\\AppData\\Local",
        "PROGRAMDATA": r"C:\\ProgramData",
        "PROGRAMFILES": r"C:\\Program Files",
        "PROGRAMFILES(X86)": r"C:\\Program Files (x86)",
        "KESTREL_API_KEY": "must-not-cross",
    }

    filtered = sanitized_subprocess_env(source)

    assert filtered == {
        key: value for key, value in source.items() if key != "KESTREL_API_KEY"
    }
