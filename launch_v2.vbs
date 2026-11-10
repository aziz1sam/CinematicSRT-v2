Option Explicit

Dim fso, scriptDir, psScript, shell, command, errMsg
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
psScript = fso.BuildPath(scriptDir, "launch_v2.ps1")

If Not fso.FileExists(psScript) Then
    errMsg = "Could not find CinematicSRT v2.0 launcher script:" & vbCrLf & psScript
    MsgBox errMsg, vbCritical, "CinematicSRT v2.0"
    WScript.Quit 1
End If

Set shell = CreateObject("WScript.Shell")
command = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & psScript & """"
shell.Run command, 0, False
