@echo off
setlocal
set MAVEN=%~dp0..\tools\apache-maven-3.9.12\bin\mvn.cmd
"%MAVEN%" -q exec:java -Dexec.mainClass=com.colastm.OrekitSmokeTest -Dorekit.data.path=orekit-data
