@echo off
rem Multi-DicomViewer launcher
rem Launch via pyw (windowed Python) detached from this console,
rem so the cmd window closes immediately and the GUI stays open.
cd /d "%~dp0"
start "" pyw run.py
