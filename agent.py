#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
import ollama  # <--- ЭТА СТРОКА ОБЯЗАТЕЛЬНА

load_dotenv()

# ---------------------------
# 1. Данные о ракетах
# ---------------------------
def load_rockets():
    df = pd.read_excel("roc.xlsx", engine="openpyxl")
    df['cost_per_kg'] = df.apply(
        lambda row: row['cost_million_usd'] * 1e6 / row['leo_capacity_kg']
        if pd.notna(row['cost_million_usd']) and row['leo_capacity_kg'] > 0
        else float('inf'), axis=1
    )
    return df

def filter_rockets(payload_kg, max_budget=None):
    df = load_rockets()
    df = df[df['leo_capacity_kg'].notna() & df['cost_million_usd'].notna()]
    df = df[df['leo_capacity_kg'] >= payload_kg]
    if max_budget:
        df = df[df['cost_million_usd'] <= max_budget]
    df = df.sort_values('cost_per_kg')
    return df.head(5).to_dict('records')

# ---------------------------
# 2. Космический мусор (оценка по высоте)
# ---------------------------
def assess_debris_risk(altitude_km):
    if altitude_km < 400:
        return "высокий (оценка по высоте)"
    elif altitude_km < 800:
        return "средний (оценка по высоте)"
    else:
        return "низкий (оценка по высоте)"

# ---------------------------
# 3. Погода через wttr.in
# ---------------------------
def get_weather_text(lat, lon):
    try:
        url = f"https://wttr.in/{lat},{lon}?format=%c+%t+%w"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return f"🌤️ {resp.text.strip()}"
        else:
            return "❌ Погода не получена"
    except:
        return "❌ Ошибка погоды"

# ---------------------------
# 4. 3D-визуализация
# ---------------------------
def generate_3d_map(altitude_km):
    R_earth = 6371
    R_orbit = R_earth + altitude_km
    np.random.seed(42)
    num = 300
    theta = np.random.uniform(0, 2*np.pi, num)
    phi = np.random.uniform(0, np.pi, num)
    x = R_orbit * np.sin(phi) * np.cos(theta)
    y = R_orbit * np.sin(phi) * np.sin(theta)
    z = R_orbit * np.cos(phi)
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    xs = R_earth * np.cos(u) * np.sin(v)
    ys = R_earth * np.sin(u) * np.sin(v)
    zs = R_earth * np.cos(v)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red'), name='Мусор (симуляция)'))
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.3, colorscale='Blues', name='Земля'))
    fig.update_layout(title=f"Симуляция мусора на высоте {altitude_km} км", scene_aspectmode='cube')
    filename = f"debris_{altitude_km}km.html"
    fig.write_html(filename)
    return filename

# ---------------------------
# 5. Извлечение параметров через Ollama + fallback
# ---------------------------
def extract_params_with_ollama(user_input):
    print("🤖 Пробую распознать через Ollama...")
    prompt = f"""
Ты – помощник. Из запроса пользователя извлеки:
- массу полезной нагрузки (в килограммах, целое число)
- высоту орбиты (в километрах, целое число)
- бюджет (в миллионах долларов США, целое число или null, если не указан)
Верни ТОЛЬКО JSON, без пояснений, без других слов.
Формат: {{"payload_kg": число, "altitude_km": число, "budget_million": число или null}}
Запрос: {user_input}
"""
    try:
        response = ollama.generate(model="qwen2.5:3b", prompt=prompt)
        text = response['response'].strip()
        print(f"📨 Ответ Ollama: {text}")
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            params = json.loads(text[start:end])
            return (params.get("payload_kg", 500), 
                    params.get("altitude_km", 500), 
                    params.get("budget_million"))
        else:
            raise ValueError("JSON не найден")
    except Exception as e:
        print(f"⚠️ Ошибка Ollama: {e}")
        print("🔁 Переключаюсь на резервный поиск цифр...")
        return extract_params_regex(user_input)

def extract_params_regex(user_input):
    numbers = re.findall(r"(\d+)", user_input)
    payload = int(numbers[0]) if len(numbers) > 0 else 500
    altitude = int(numbers[1]) if len(numbers) > 1 else 500
    budget_match = re.search(r"(\d+)\s*(?:млн|миллионов?)", user_input, re.IGNORECASE)
    budget = int(budget_match.group(1)) if budget_match else None
    return payload, altitude, budget

# ---------------------------
# 6. Главный цикл
# ---------------------------
last_altitude = 500

def agent():
    global last_altitude
    print("🚀 AI-агент по выбору ракеты-носителя (треки A+C)")
    print("Погода: wttr.in | Оценка мусора: по высоте")
    print("Данные о ракетах: roc.xlsx")
    print("Пример: 'запусти 300 кг на 550 км, бюджет 10 млн'")
    print("Для 3D-карты после запроса: 'покажи карту'")
    print("🤖 Использую локальную LLM (Ollama, qwen2.5:3b) с запасным парсингом цифр.\n")

    while True:
        user_input = input("Ваш запрос (или 'выход'): ").strip()
        if user_input.lower() in ("выход", "quit", "exit"):
            break

        if "карту" in user_input.lower() or "карта" in user_input.lower():
            if last_altitude:
                file = generate_3d_map(last_altitude)
                print(f"🌍 3D-карта мусора (высота {last_altitude} км) сохранена: {file}")
            else:
                print("⚠️ Сначала задайте высоту в запросе")
            continue

        payload_kg, altitude_km, max_budget = extract_params_with_ollama(user_input)
        last_altitude = altitude_km
        print(f"📊 Масса: {payload_kg} кг, высота: {altitude_km} км, бюджет: {max_budget if max_budget else 'не ограничен'} млн $")

        risk = assess_debris_risk(altitude_km)
        print(f"🧹 Риск столкновения с мусором: {risk}")

        rockets = filter_rockets(payload_kg, max_budget)
        if not rockets:
            print("❌ Нет ракет под ваши параметры.")
        else:
            print(f"✅ Топ-{len(rockets)} ракет (по стоимости за кг):")
            for i, r in enumerate(rockets, 1):
                weather = get_weather_text(r['lat'], r['lon'])
                print(f"\n{i}. {r['name']} ({r['country']}) — {r['cost_million_usd']} млн $, Грузоподъёмность {r['leo_capacity_kg']} кг")
                print(f"   Надёжность: {r['success_rate']*100:.1f}% | Космодром: {r['launch_site']}")
                print(f"   Погода: {weather}")
            print("\n💡 Чтобы увидеть 3D-карту мусора, напишите 'покажи карту'.")
        print("-" * 50)

if __name__ == "__main__":
    if not os.path.exists("roc.xlsx"):
        print("❌ Файл roc.xlsx не найден!")
    else:
        agent()