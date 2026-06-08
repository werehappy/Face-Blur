; FACEBLUR Inno Setup Script
; Detects GPU during install and installs correct version

#define AppName "FACEBLUR"
#define AppVersion "1.0.0"
#define AppPublisher "werehappy"
#define AppExeName "FACEBLUR.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={userdesktop}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=installer_output
OutputBaseFilename=FACEBLUR_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "dist\FACEBLUR_CPU.exe"; DestDir: "{app}"; DestName: "FACEBLUR.exe"; Flags: ignoreversion; Check: not HasNvidiaGPU
Source: "dist\FACEBLUR_GPU.exe"; DestDir: "{app}"; DestName: "FACEBLUR.exe"; Flags: ignoreversion; Check: HasNvidiaGPU
Source: "ffmpeg.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  GPUDetected: Boolean;
  GPUChecked: Boolean;

function FindNvidiaSmi: String;
var
  Paths: TArrayOfString;
  I: Integer;
begin
  Result := '';
  // Common nvidia-smi locations
  SetArrayLength(Paths, 5);
  Paths[0] := 'C:\Windows\System32\nvidia-smi.exe';
  Paths[1] := 'C:\Windows\SysWOW64\nvidia-smi.exe';
  Paths[2] := 'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe';
  Paths[3] := ExpandConstant('{pf}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');
  Paths[4] := ExpandConstant('{pf32}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');

  for I := 0 to GetArrayLength(Paths) - 1 do
  begin
    if FileExists(Paths[I]) then
    begin
      Result := Paths[I];
      Exit;
    end;
  end;
end;

function HasNvidiaGPU: Boolean;
var
  ResultCode: Integer;
  TempFile: String;
  Output: TArrayOfString;
  NvidiaSmi: String;
begin
  if GPUChecked then
  begin
    Result := GPUDetected;
    Exit;
  end;

  GPUChecked := True;
  GPUDetected := False;
  TempFile := ExpandConstant('{tmp}\gpu_check.txt');

  // Find nvidia-smi
  NvidiaSmi := FindNvidiaSmi;

  if NvidiaSmi <> '' then
  begin
    // Run nvidia-smi directly with full path
    if Exec(NvidiaSmi, '-L > "' + TempFile + '"', '',
            SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      if ResultCode = 0 then
      begin
        if LoadStringsFromFile(TempFile, Output) then
        begin
          if GetArrayLength(Output) > 0 then
            GPUDetected := True;
        end;
      end;
    end;
  end else
  begin
    // Try via cmd as fallback
    if Exec('cmd.exe', '/c nvidia-smi -L > "' + TempFile + '" 2>&1', '',
            SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      if ResultCode = 0 then
      begin
        if LoadStringsFromFile(TempFile, Output) then
        begin
          if GetArrayLength(Output) > 0 then
            GPUDetected := True;
        end;
      end;
    end;
  end;

  // Also check for NVIDIA registry key as extra fallback
  if not GPUDetected then
  begin
    if RegKeyExists(HKLM, 'SOFTWARE\NVIDIA Corporation\Global') then
      GPUDetected := True;
    if RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\NVIDIA Corporation\Global') then
      GPUDetected := True;
  end;

  Result := GPUDetected;
end;

procedure InitializeWizard;
begin
  GPUChecked := False;
  GPUDetected := False;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  S: String;
begin
  S := '';
  if MemoDirInfo <> '' then S := S + MemoDirInfo + NewLine + NewLine;
  if MemoGroupInfo <> '' then S := S + MemoGroupInfo + NewLine + NewLine;
  if MemoTasksInfo <> '' then S := S + MemoTasksInfo + NewLine + NewLine;

  if HasNvidiaGPU then
    S := S + 'Version: GPU-accelerated (Nvidia GPU detected)' + NewLine
  else
    S := S + 'Version: CPU only (no Nvidia GPU detected)' + NewLine;

  Result := S;
end;
