; FACEBLUR Inno Setup Script
; Ships ONE small exe (torch excluded) plus a bundled embeddable Python.
; torch is downloaded on the user's first launch, sized to their GPU.

#define AppName "FACEBLUR"
#define AppVersion "1.2.1"
#define AppPublisher "werehappy"
#define AppExeName "FACEBLUR.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; Per-user install location, writable without admin so first-run torch
; install can create faceblur_env\Lib\site-packages here.
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
OutputDir=installer_output
OutputBaseFilename=FACEBLUR_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Always show the "Select Destination Location" page. Without this, Inno's
; default (DisableDirPage=auto) hides it on upgrades when a prior install with
; the same AppId is detected, silently reusing the old folder.
DisableDirPage=no
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Single exe (torch is NOT bundled - downloaded at first run)
Source: "dist\FACEBLUR.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ffmpeg.exe"; DestDir: "{app}"; Flags: ignoreversion
; Trained head detector. Ships next to the exe so the app's "user head.pt
; wins" path picks it up (no _MEIPASS / no rebuild needed to swap it later).
Source: "head.pt"; DestDir: "{app}"; Flags: ignoreversion
; Bundled embeddable Python (pip-capable, no torch yet). recursesubdirs
; copies the whole tree; torch lands inside it on first run.
Source: "faceblur_env\*"; DestDir: "{app}\faceblur_env"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; torch is installed into faceblur_env after setup, so remove the whole tree
; (and any first-run debug log) on uninstall.
Type: filesandordirs; Name: "{app}\faceblur_env"
Type: files; Name: "{app}\faceblur_debug.txt"

[Messages]
; Friendly heads-up shown on the final page.
FinishedLabel=Setup is complete.%n%nThe first time you open {#AppName}, it will download the libraries it needs for your hardware (a one-time setup that needs an internet connection). After that, it starts instantly.
