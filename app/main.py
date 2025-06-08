import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

import os
import json
import requests
from ultralytics import YOLO
import torch
import timm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.ops import nms
# from google.colab import drive # Colab 환경이 아니라면 이 부분은 주석 처리하거나 로컬 환경에 맞게 수정

# 팀원 1 코드에서 필요한 라이브러리
import unicodedata
import difflib
import pandas as pd

# ✅ Groq API 설정
GROQ_API_KEY = "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" # 실제 사용 시 안전하게 관리하세요.
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ✅ 경로 설정 (로컬 환경에 맞게 수정 필요)
# 예시 경로입니다. 실제 파일 위치에 맞게 수정하세요.
# Colab을 사용하지 않는 경우, Google Drive 마운트 없이 직접 경로를 지정해야 합니다.
base_path = "./" # 현재 스크립트가 있는 폴더를 기본 경로로 가정
image_path = os.path.join(base_path, "IMG_9906.JPG")  # 예시 이미지 파일명
model_path = os.path.join(base_path, "main6.pt")
class_idx_path = os.path.join(base_path, "class_to_idx.json")
nutrition_db_path = os.path.join(base_path, "simplified_fooddata.json") # 팀원 1의 영양소 DB
font_path = os.path.join(base_path, "NanumGothic.ttf") # 나눔고딕 폰트 경로 (시각화용)

#Google Drive 마운트 (Colab 환경에서 필요시 주석 해제)
try:
    # Colab 사용 시 경로 재설정 예시 (실제 파일 위치에 맞게 수정)
    base_path = "/content/drive/MyDrive/"
    image_path = os.path.join(base_path, "IMG_9906.JPG")
    model_path = os.path.join(base_path, "main6.pt")
    class_idx_path = os.path.join(base_path, "class_to_idx.json")
    nutrition_db_path = os.path.join(base_path, "simplified_fooddata.json")
    font_path = os.path.join(base_path, "NanumGothic.ttf")
    restaurant_path = os.path.join(base_path, "식당_메뉴_정규화_템플릿.csv")
except Exception as e:
    print(f"Google Drive 마운트 중 오류 발생 (Colab 환경이 아닐 경우 무시): {e}")


# 1. 사용자로부터 정보 입력 받기
print("사용자 정보를 입력해주세요. 각 항목 입력 후 Enter를 누르세요.")
user_data = {}
try:
    name = input("이름 (name): ")
    height = float(input("키(cm): "))
    weight = float(input("몸무게(kg): "))
    while True:
        age_str = input("나이 (age): ")
        if age_str.isdigit():
            age = int(age_str)
            break
        else:
            print("나이는 숫자로 입력해주세요.")
    activity_level = input("활동량 (low, medium, high): ").lower()
    food_preference = input("음식 선호도 (예: 고단백, 채식, 없음): ") or "없음"

    if activity_level == "low":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif activity_level == "medium":
        bmr = 10 * weight + 6.25 * height - 5 * age + 10
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 15

    calories_needed = bmr * 1.2 if activity_level == "low" else (bmr * 1.55 if activity_level == "medium" else bmr * 1.9)
    print(f"하루 필요 칼로리: {calories_needed:.0f} kcal")

    user_data = {
        "name": name, "height": height, "weight": weight, "age": age,
        "activity_level": activity_level, "food_preference": food_preference,
        "bmr": bmr, "calories_needed": calories_needed
    }

    #사용자 정보 JSON 파일로 저장 (선택적)
    file_name = "user_info_from_input.json"
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(user_data, json_file, ensure_ascii=False, indent=4)
    print(f"사용자 정보가 {file_name}에 저장되었습니다.")

except Exception as e:
    print(f"\n사용자 정보 입력 중 오류가 발생했습니다: {e}")
    # 오류 발생 시 프로그램 중단 또는 기본값 사용 등의 처리 필요
    user_data = {"calories_needed": 2000, "food_preference": "상관없음"} # 예시 기본값

# ✅ class_to_idx 복원 → class_names 리스트 재구성
try:
    with open(class_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    class_names = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        class_names[idx] = name
    num_classes = len(class_names)
except Exception as e:
    print(f"클래스 정보 로드 중 오류: {e}. 일부 기능이 제한될 수 있습니다.")
    class_names = [] # 빈 리스트로 초기화
    num_classes = 0

# ✅ 이미지 로드
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"이미지 로드 중 오류: {e}. 이미지 분석을 진행할 수 없습니다.")
    image = None # 이미지 로드 실패 처리

predicted_foods = []
if image and class_names: # 이미지와 클래스 정보가 모두 로드된 경우에만 실행
    # ✅ YOLO 탐지
    yolo_model = YOLO("yolov5s.pt") # 사전 학습된 모델 사용, 필요시 경로 지정
    results = yolo_model(image)

    # ✅ NMS 중복 제거 (결과가 리스트 형태일 수 있으므로 첫 번째 요소 사용)
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        keep_indices = nms(boxes, scores, iou_threshold=0.4)
        filtered_boxes = boxes[keep_indices]
    else:
        filtered_boxes = torch.empty((0, 4)) # 탐지된 객체가 없을 경우 빈 텐서

    # ✅ EfficientNet 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        eff_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
        eff_model.load_state_dict(torch.load(model_path, map_location=device))
        eff_model = eff_model.to(device)
        eff_model.eval()

        # ✅ 전처리
        transform_eff = transforms.Compose([ # 변수명 변경 (기존 'transform'과 충돌 방지)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # ✅ 이미지 시각화 준비
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        try:
            font = ImageFont.truetype(font_path, size=22)
        except IOError:
            print(f"폰트 파일({font_path})을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            font = ImageFont.load_default()


        # ✅ 추론 실행
        drawn_labels = set()
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = image.crop((x1, y1, x2, y2))
            input_tensor = transform_eff(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = eff_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            label = class_names[pred] if pred < len(class_names) else f"Unknown({pred})"
            if label not in drawn_labels: # 중복 클래스 시각화 및 추가 방지
                drawn_labels.add(label)
                predicted_foods.append(label)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 25), label, fill="red", font=font) # 텍스트 위치 조정

        # ✅ 이미지 출력
        plt.figure(figsize=(12, 8))
        plt.imshow(draw_image)
        plt.axis("off")
        plt.title("음식 인식 결과 (YOLO + EfficientNet)")
        plt.show()

    except Exception as e:
        print(f"EfficientNet 모델 처리 중 오류: {e}")

else:
    if not image:
        print("이미지가 로드되지 않아 음식 인식을 건너뜁니다.")
    if not class_names:
        print("클래스 정보가 없어 음식 인식을 건너뜁니다.")


# --- 팀원 1 작업 통합: 영양소 정보 매핑 ---
nutrition_infos = []
if predicted_foods: # 인식된 음식이 있을 경우에만 실행
    try:
        with open(nutrition_db_path, encoding="utf-8") as f:
            simplified_db = json.load(f)

        print("\n--- 인식된 음식 영양 정보 매핑 ---")
        for label in predicted_foods:
            label_norm = unicodedata.normalize("NFC", label) # 유니코드 정규화
            match = None
            mapped_key_name = label_norm # 매칭된 DB의 키 이름 (기본값은 원본 레이블)

            # 1. 정확 일치
            if label_norm in simplified_db:
                match = simplified_db[label_norm]
                print(f"✅ '{label_norm}' (정확히 매핑됨)")
            else:
                # 2. 부분 포함 기반 후보 탐색
                candidates = []
                for key_db in simplified_db:
                    key_db_norm = unicodedata.normalize("NFC", key_db)
                    if label_norm in key_db_norm or key_db_norm in label_norm:
                        candidates.append(key_db) # 원본 DB키 저장

                if candidates:
                    best_candidate = candidates[0] # 간단히 첫번째 후보 사용
                    close_candidates = difflib.get_close_matches(label_norm, candidates, n=1, cutoff=0.7)
                    if close_candidates:
                        best_candidate = close_candidates[0]

                    match = simplified_db[best_candidate]
                    mapped_key_name = best_candidate
                    print(f"🔁 '{label_norm}' ≈ '{best_candidate}' (부분/유사도 매핑됨)")
                else:
                    # 3. difflib 유사도 기반 후보 탐색 (부분 포함 실패 시)
                    close_matches = difflib.get_close_matches(label_norm, simplified_db.keys(), n=1, cutoff=0.6) # cutoff 조절 가능
                    if close_matches:
                        match_key = close_matches[0]
                        match = simplified_db[match_key]
                        mapped_key_name = match_key
                        print(f"🧠 '{label_norm}' ≈ '{match_key}' (유사도 매칭됨)")
                    else:
                        print(f"❌ '{label_norm}' (매핑 실패)")

            if match:
                nutrition_infos.append({
                    "food": mapped_key_name, # DB에서 매칭된 음식 이름 사용
                    "kcal": match.get("kcal", "0"),
                    "carbs": match.get("carbs", "0"),
                    "protein": match.get("protein", "0"),
                    "fat": match.get("fat", "0")
                })
            else: # 매칭 실패 시
                nutrition_infos.append({
                    "food": label_norm, # 원본 인식 이름 사용
                    "kcal": "N/A", "carbs": "N/A", "protein": "N/A", "fat": "N/A"
                })

        print("---------------------------------")

    except FileNotFoundError:
        print(f"영양소 DB 파일({nutrition_db_path})을 찾을 수 없습니다. 영양 정보 없이 진행합니다.")
    except Exception as e:
        print(f"영양 정보 처리 중 오류: {e}")
else:
    print("\n인식된 음식이 없어 영양 정보 매핑을 건너뜁니다.")

restaurant_menu_for_prompt = "학교 근처 식당 메뉴 정보를 불러오지 못했습니다. 일반적인 메뉴 중에서 추천합니다." # 기본 메시지
try:
    if not os.path.exists(restaurant_path):
        print(f"경고: 식당 메뉴 CSV 파일({restaurant_path})을 찾을 수 없습니다. AI가 일반 지식을 활용하여 추천합니다.")
    else:
        try:
          df_restaurants = pd.read_csv(restaurant_path, encoding='cp949')
        except UnicodeDecodeError:
      
          print(f"'{restaurant_path}' 파일을 'utf-8' 및 'cp949'로 읽는데 실패했습니다. 파일 인코딩을 확인해주세요.")
          
        except Exception as e:
          print(f"식당 메뉴 CSV 처리 중 예상치 못한 오류: {e}. AI가 일반 지식을 활용하여 추천합니다.")



    menu_items_for_prompt = []
    for index, row in df_restaurants.head(15).iterrows():
      restaurant_name = row.get('식당명', '')  
      normalized_menu = row.get('정규화된이름', '') 
      
      kcal = row.get('에너지(kcal)', '정보없음')
      kcal_str = str(row.get("에너지(kcal)", "0"))
    carbs_str = str(row.get("탄수화물(g)", "0"))
    protein_str = str(row.get("단백질(g)", "0"))
    fat_str = str(row.get("지방(g)", "0"))

      # ... (탄수화물, 단백질, 지방 정보 가져오는 코드) ...
      # kcal_str, carbs_str 등 문자열로 변환하는 코드 필요

      # 표시될 메뉴 이름 조합
    display_name = f"{restaurant_name} - {normalized_menu}" if restaurant_name else normalized_menu

    menu_items_for_prompt.append(
         f"- {display_name}: 약 {kcal_str}kcal (탄수화물 약 {carbs_str}g, 단백질 약 {protein_str}g, 지방 약 {fat_str}g)"
    )

    if menu_items_for_prompt:
            restaurant_menu_for_prompt = "다음은 참고할 수 있는 학교 근처 식당 메뉴 목록입니다 (CSV 기반):\n" + "\n".join(menu_items_for_prompt)
    else:
            restaurant_menu_for_prompt = "학교 근처 식당 메뉴 CSV를 읽었으나, 프롬프트에 포함할 유효한 메뉴 정보를 추출하지 못했습니다. 일반적인 메뉴 중에서 추천합니다."

except FileNotFoundError:
    print(f"식당 메뉴 CSV 파일({restaurant_path})을 찾을 수 없습니다. AI가 일반 지식을 활용하여 추천합니다.")
except pd.errors.EmptyDataError:
    print(f"식당 메뉴 CSV 파일({restaurant_path})이 비어있습니다. AI가 일반 지식을 활용하여 추천합니다.")
except Exception as e:
    print(f"식당 메뉴 CSV 처리 중 오류: {e}. AI가 일반 지식을 활용하여 추천합니다.")


# ✅ 통합 프롬프트 생성
# 섭취한 음식 영양 정보 문자열 생성
if nutrition_infos:
    eaten_foods_nutrition_str = "\n".join([
        f"- {n['food']}: 약 {n['kcal']}kcal (탄수화물 {n['carbs']}g, 단백질 {n['protein']}g, 지방 {n['fat']}g)"
        for n in nutrition_infos if n['kcal'] != "N/A" # 매핑된 정보만 포함
    ])
    if not eaten_foods_nutrition_str: # 모든 음식이 N/A 인 경우
         eaten_foods_nutrition_str = "인식된 음식의 구체적인 영양 정보를 찾지 못했습니다."
    eaten_foods_summary_for_prompt = f"최근 식사로 다음 음식들을 섭취했습니다:\n{eaten_foods_nutrition_str}"
else:
    eaten_foods_summary_for_prompt = "최근 섭취한 음식에 대한 정보가 없거나 이미지에서 음식을 인식하지 못했습니다."




prompt = f"""
###역할###
당신은 사용자의 건강 데이터, 최근 섭취한 음식의 영양 정보, 그리고 아래에 제공된 학교 근처 식당 메뉴 목록을 종합적으로 분석하여, 개인 맞춤형 식단을 추천하는 AI 영양사입니다. 당신의 작업은 다음 끼니에 적합하며 영양학적으로 균형 잡힌 식단 옵션을 구체적으로 제안하는 것입니다.

###사용자 정보###
- 하루 권장 섭취 칼로리: {user_data.get('calories_needed', 2000.0):.0f} kcal
- 기타 개인적인 고려사항: (현재는 제공되지 않았습니다. 필요시 AI가 질문할 수 있습니다.)

###최근 섭취 음식 분석 요청###
{eaten_foods_summary_for_prompt}
위 섭취 내용을 바탕으로 다음 사항을 먼저 계산하고 간략히 분석해주십시오:
1. 섭취한 음식들의 총 예상 칼로리
2. 섭취한 음식들의 총 탄수화물, 단백질, 지방의 양(g)
3. 현재까지의 영양 균형 상태 (예: 탄수화물 위주 섭취, 단백질 부족 등)

###학교 근처 식당 메뉴 (CSV 제공)###
{restaurant_menu_for_prompt}

###다음 끼니 추천 요청###
위 분석과 사용자 정보, 그리고 **위에 제시된 '학교 근처 식당 메뉴 (CSV 제공)' 목록만으로**, 다음 끼니로 섭취할 만한 식단 옵션을 3가지 추천해라.
- **중요**: 사용자의 하루 권장 섭취 칼로리 ({user_data.get('calories_needed'):.0f}kcal)를 고려하여 식단 추천.
- **식당 메뉴 고려**: 사용자의 영양 균형을 고려하여, **위에 제시된 '학교 근처 식당 메뉴 (CSV 제공)' 목록 중에서** 가장 적절한 옵션이나 현실적인 조합을 찾아 추천해라. 만약 목록에 적절한 메뉴가 없거나 사용자의 선호와 맞지 않을 경우에만, 일반적인 외식 또는 직접 조리 가능한 메뉴를 대안으로 제시할 수 있습니다.
- 각 식단 옵션은 다음 정보를 반드시 포함해야 합니다:
    a. 추천하는 전체 식단 또는 메뉴  (예: 음식점 : 음식1 과 음식2, 음시점 : 음식1, 음식점 : 음식1과 음식2와 음식3)
    b. 각 음식의 구체적인 종류 및 권장 섭취량 (제시된 메뉴 정보를 바탕으로)
    c. 해당 옵션의 예상 총 섭취 칼로리
    d. 주요 영양소(탄수화물, 단백질, 지방)의 대략적인 구성 비율 또는 양(g) (제시된 메뉴 정보를 최대한 활용하며, 정보가 부족한 경우 추정치임을 명시할 수 있음)
    e. 해당 식단을 추천하는 간결한 이유 (예: 부족한 단백질 보충, 부족한 영양성분 보충 등)

###출력 조건###
- 모든 답변은 반드시, 그리고 예외 없이 한국어로만 생성해야 합니다. 다른 언어 사용은 절대 금지됩니다.
- 전문적인 용어 사용은 최소화하고, 일반인이 쉽게 이해할 수 있도록 명확하고 간결하게 설명해주십시오.
- 최근 섭취 음식 분석 결과와 다음 끼니 추천을 명확히 구분하여 제시해주십시오.
- 추천 식단 옵션은 번호 등으로 구분하고, 각 옵션 내 정보도 명확하게 나열해주십시오.

###추가 정보 요청 지침###
만약 위 정보 외에 식단 추천 및 분석을 위해 사용자에게 반드시 필요한 추가 정보(예: 특정 알레르기, 다음 식사 시간대 등)가 있다면, 분석 및 추천 전에 사용자에게 명확히 질문해주십시오.
"""

# ✅ Groq API 호출
print("\n--- Groq API 호출 시작 ---")
try:
    response = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-8b-8192", 
            "messages": [
                {
                    "role": "system",
                    "content": "너는 사용자의 건강 정보, 최근 섭취 음식의 영양 구성을 종합적으로 분석하여 다음 끼니를 추천하는 전문 AI 영양사다. 모든 답변은 반드시 한국어로만 명확하고 간결하게 제공해야 하며, 다른 언어 사용은 절대 허용되지 않는다. 사용자의 요청 사항을 단계별로 충실히 수행하라."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1 # 일관된 답변을 위해 낮은 값 설정
        }
    )
    response.raise_for_status() # 오류 발생 시 예외 발생

    res_json = response.json()
    print("Groq API 응답 상태:", response.status_code)
    if "choices" in res_json and res_json["choices"]:
        print("\n🍱 AI 영양사 식단 추천:")
        print(res_json["choices"][0]["message"]["content"])
    else:
        print("❗ API 응답에 'choices'가 없거나 비어있습니다:")
        print(res_json)

except requests.exceptions.RequestException as e:
    print(f"❗ Groq API 호출 중 오류 발생: {e}")
except Exception as e:
    print(f"❗ 결과 처리 중 알 수 없는 오류 발생: {e}")
    if 'response' in locals(): # 응답 객체가 있다면 내용 출력
        print("오류 발생 시 API 응답 내용:", response.text)
