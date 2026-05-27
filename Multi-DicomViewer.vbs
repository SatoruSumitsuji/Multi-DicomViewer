' Multi-DicomViewer launcher (no console window at all)
' Double-click this file to launch the GUI without any terminal flashing.
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = scriptDir
' 0 = hidden window, False = do not wait for the process to finish
shell.Run "pyw run.py", 0, False
