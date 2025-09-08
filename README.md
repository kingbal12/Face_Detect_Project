# Face Detect Project

## 📌 프로젝트 개요
이 프로젝트는 단순히 얼굴을 탐지하는 데 그치지 않고, **얼굴을 구역별로 분할하여 피부 상태를 정량적으로 분석**하는 백엔드 서비스입니다.  
주요 목표는 얼굴 각 영역(이마, 눈가, 볼 등)의 **주름·음영을 측정하여 주름 점수를 산출**하고, 피부의 **RGB 값을 기반으로 미백 점수를 계산**하는 것입니다.  
이를 통해 피부 상태를 객관적으로 수치화하고, 뷰티/헬스케어 서비스에 활용할 수 있도록 설계되었습니다.  

## 🛠 사용 기술
- **Backend**: Python, Flask (또는 FastAPI)  
- **영상 처리**: OpenCV  
- **알고리즘**: Histogram Equalization, Edge Detection, Contour Analysis  
- **색상 분석**: RGB 채널 기반 평균값/분산 분석  
- **Database**: MySQL (피부 점수 및 사용자 데이터 저장)  
- **기타**: RESTful API, Git, Github  

## ✨ 주요 기능
1. **얼굴 영역 분할 (Face Region Segmentation)**  
   - 얼굴을 이마, 눈가, 볼, 입 주변 등 구역별로 나누어 분석  

2. **주름 분석 (Wrinkle Score)**  
   - Edge Detection(예: Canny) 및 음영 분석을 통해 주름 강도 계산  
   - 구역별 점수를 산출 후 종합 주름 점수 제공  

3. **미백 분석 (Whitening Score)**  
   - 피부 RGB 값을 추출하여 평균 밝기(Lightness) 및 채널 비율 계산  
   - 구역별 미백 점수 및 종합 점수 산출  

4. **데이터 저장 및 API 제공**  
   - REST API를 통해 클라이언트에 JSON 형태로 결과 제공  

## 🚀 실행 방법
1. 저장소 클론
   ```bash
   git clone https://github.com/kingbal12/Face_Detect_Project.git
