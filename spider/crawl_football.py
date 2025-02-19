import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
import os
import pandas as pd
from tqdm import tqdm

def generate_dates(start_date, end_date):
    """生成日期范围"""
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

def fetch_daily_data(date):
    """爬取单日数据"""
    url = f"https://www.okooo.com/jingcai/{date}/"
    html = fetch_html(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    matches = soup.find_all('div', class_='touzhu_1')
    
    data = []
    for match in matches:
        # 提取日期 --------------------------------------------------
        time_div = match.find('div', class_='shijian')
        if time_div and time_div.has_attr('title'):
            full_time = time_div['title'].split(': ')[-1]  # 格式 "2025-02-18 20:00:00"
            date = full_time.split()[0]  # 提取日期部分 "2025-02-18"
        else:
            date = "N/A"
        
        # 判断是否截止 ----------------------------------------------
        is_ended = "已截止" if match.get('data-end') == '1' else "未截止"
        
        # 提取主队和客队名称 -----------------------------------------
        home = match.find('div', class_='zhu').find(class_='zhum').text.strip()
        away = match.find('div', class_='fu').find(class_='zhum').text.strip()
        
        # 提取胜负赔率 -----------------------------------------------
        win = match.find('div', class_='zhu').find(class_='peilv').text.strip()
        draw = match.find('div', class_='ping').find(class_='peilv').text.strip()
        lose = match.find('div', class_='fu').find(class_='peilv').text.strip()
        
        # 提取让球盘信息 ---------------------------------------------
        rq_div = match.find('div', class_='rangqiuspf')
        if rq_div:
            rq_span = rq_div.find('span', class_='rangqiu')
            rq_num = rq_span.text.strip() if rq_span else "N/A"
            
            rq_zhu_elem = rq_div.find('div', class_='zhu')
            rq_zhu = rq_zhu_elem.find(class_='peilv').text.strip() if rq_zhu_elem else "N/A"
            
            rq_ping_elem = rq_div.find('div', class_='ping')
            rq_ping = rq_ping_elem.find(class_='peilv').text.strip() if rq_ping_elem else "N/A"
            
            rq_fu_elem = rq_div.find('div', class_='fu')
            rq_fu = rq_fu_elem.find(class_='peilv').text.strip() if rq_fu_elem else "N/A"
        else:
            rq_num = rq_zhu = rq_ping = rq_fu = "N/A"
        
        # 提取赛果 --------------------------------------------------
        saiguo_elem = match.find('div', class_='more_bg')
        saiguo = saiguo_elem.find('p', class_='p1').text.strip() if saiguo_elem else "N/A"
        
        data.append({
            '比赛': f"{home} vs {away}",
            '日期': date,
            '是否截止': is_ended,
            '主胜': win,
            '平局': draw,
            '客胜': lose,
            '让球盘': rq_num,
            '让球主胜': rq_zhu,
            '让球平': rq_ping,
            '让球主负': rq_fu,
            '赛果': saiguo
        })
    return data

def save_to_csv(data, date):
    """保存单日数据到CSV"""
    filename = f"data/足球_{date}.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

def main():
    # 配置参数
    start_date = "2022-02-17"
    end_date = "2025-02-18"
    output_file = "足球大表.csv"
    
    # 生成日期列表
    dates = generate_dates(start_date, end_date)
    
    # 爬取数据
    all_data = []
    for date in tqdm(dates, desc="总进度"):
        # 爬取单日数据
        daily_data = fetch_daily_data(date)
        
        # 保存当日数据
        if daily_data:
            save_to_csv(daily_data, date)
            all_data.extend(daily_data)
        
        # 延迟防止封IP
        time.sleep(0.5)
    
    # 合并所有数据
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据已合并到 {output_file}")

def fetch_html(url):
    """带重试机制的请求函数"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    retries = 3
    for _ in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"请求失败 {url}: {str(e)}")
            time.sleep(0.1)
    return None

if __name__ == "__main__":
    main()
