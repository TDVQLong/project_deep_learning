@echo off
chcp 65001 > nul
title PACS AI - Chẩn đoán Gãy xương Cổ tay Trẻ em
echo =====================================================================
echo    KHỞI ĐỘNG HỆ THỐNG PACS AI CHẨN ĐOÁN GÃY XƯƠNG CỔ TAY (YOLOv8)
echo =====================================================================
echo.
echo Đang kiểm tra môi trường và khởi động giao diện Web...
streamlit run app.py
pause
