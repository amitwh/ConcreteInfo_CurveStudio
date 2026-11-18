; Inno Setup script for ConcreteInfo CurveStudio (built by CI, best effort).
;
;   iscc /DMyAppVersion=<version> apps\desktop\packaging\windows\curvestudio.iss
;
; run from the repository root after apps/desktop/packaging/build.py produced
; dist\curvestudio-desktop and build\icons. All relative paths resolve against
; this file's directory. The output lands in dist\release next to the portable
; zip, so the CI "sums" step picks it up.

#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{9AB80E5F-6B23-4D1D-8B0F-CBD0C2F3A111}
AppName=ConcreteInfo CurveStudio
AppVersion={#MyAppVersion}
AppPublisher=ConcreteInfo
DefaultDirName={autopf}\CurveStudio
DefaultGroupName=ConcreteInfo CurveStudio
UninstallDisplayIcon={app}\curvestudio-desktop.exe
OutputDir=..\..\..\dist\release
OutputBaseFilename=curvestudio-desktop-setup
SetupIconFile=..\..\..\build\icons\curvestudio-desktop.ico
LicenseFile=..\..\..\..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Files]
Source: "..\..\..\dist\curvestudio-desktop\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ConcreteInfo CurveStudio"; Filename: "{app}\curvestudio-desktop.exe"
Name: "{autoprograms}\ConcreteInfo CurveStudio"; Filename: "{app}\curvestudio-desktop.exe"
Name: "{autodesktop}\ConcreteInfo CurveStudio"; Filename: "{app}\curvestudio-desktop.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\curvestudio-desktop.exe"; Description: "Launch ConcreteInfo CurveStudio"; Flags: nowait postinstall skipifsilent
