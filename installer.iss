; Inno Setup script for Filter Annotation Refiner.
; Driven by build.ps1, which passes MyAppVersion and MySourceDir so the version
; comes from refiner\version.py and is never typed twice.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef MySourceDir
  #define MySourceDir "build\app.dist"
#endif

#define MyAppName "Filter Annotation Refiner"
#define MyAppExeName "FilterAnnotationRefiner.exe"

[Setup]
AppId={{8F3A2C14-5B7D-4E29-9A61-6C4D8E2F7B30}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=ITRI
DefaultDirName={autopf}\FilterAnnotationRefiner
DefaultGroupName={#MyAppName}
OutputDir=dist
OutputBaseFilename=FilterAnnotationRefiner_Setup_v{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
; The bundle carries CUDA torch, so 32-bit targets cannot run it.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
DisableProgramGroupPage=yes
LicenseFile={#MySourceDir}\LICENSE
WizardStyle=modern
; Several GB of model weights and CUDA libraries.
DiskSpanning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesetrad"; MessagesFile: "compiler:Languages\ChineseTraditional.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// The bundle ships CUDA torch. Without an NVIDIA driver the app still starts but
// falls back to CPU and is unusably slow, so warn at install time rather than
// letting the operator discover it mid-conversion.
function InitializeSetup(): Boolean;
var
  DriverPath: String;
begin
  Result := True;
  DriverPath := ExpandConstant('{sys}\nvml.dll');
  if not FileExists(DriverPath) then
    Result := MsgBox(
      'No NVIDIA driver was detected on this machine.' + #13#10#13#10 +
      'The application will run on CPU instead, which is many times slower ' +
      '(minutes per image rather than under a second).' + #13#10#13#10 +
      'Continue anyway?', mbConfirmation, MB_YESNO) = IDYES;
end;
