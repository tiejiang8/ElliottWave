# 步骤1: 数据获取和预处理模块
def get_stock_data(symbol, start_date=None, end_date=None):
    """下载股票数据"""
    import akshare as ak
    
    current_dir = os.getcwd()
    today = date.today().strftime("%Y%m%d")
    os.makedirs(os.path.join(current_dir, 'talibdata'), exist_ok=True)
    
    csv_file_path = os.path.join(current_dir, 'talibdata', f'{symbol}_1d_{today}.csv')
    
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        try:
            if symbol == 'GC':
                df = ak.futures_foreign_hist("GC")
            else:
                df = ak.stock_us_daily(symbol=symbol, adjust='qfq')
            df.to_csv(csv_file_path, index=False)
        except Exception as e:
            print(f"下载数据出错: {str(e)}")
            files = [f for f in os.listdir(os.path.join(current_dir, 'talibdata')) 
                    if f.startswith(f'{symbol}_1d_')]
            if files:
                latest_file = max(files)
                csv_file_path = os.path.join(current_dir, 'talibdata', latest_file)
                print(f"使用本地缓存数据: {latest_file}")
                df = pd.read_csv(csv_file_path)
            else:
                print(f"未找到 {symbol} 的历史数据")
                return pd.DataFrame()
    
    # 转换日期并筛选时间范围
    df['date'] = pd.to_datetime(df['date'])
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]
        
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]
    
    return df


# 步骤2: 极值点检测模块 - 单独测试验证
def detect_extrema(prices, dates, method='adaptive', window_range=(5, 20)):
    """
    检测价格序列中的极值点
    
    参数:
        prices: 价格序列
        dates: 对应的日期序列
        method: 检测方法 ('simple', 'adaptive', 'multi_scale')
        window_range: 窗口大小范围(min_window, max_window)
    
    返回:
        极值点列表，每个点包含 (index, date, price, type)
    """
    extrema = []
    min_window, max_window = window_range
    
    if method == 'simple':
        # 简单固定窗口方法
        for window in range(min_window, max_window+1, 5):
            maxima = signal.argrelextrema(prices, np.greater, order=window)[0]
            minima = signal.argrelextrema(prices, np.less, order=window)[0]
            
            for idx in maxima:
                extrema.append((idx, dates[idx], prices[idx], 1))
            
            for idx in minima:
                extrema.append((idx, dates[idx], prices[idx], -1))
    
    elif method == 'adaptive':
        # 自适应窗口方法
        volatility = pd.Series(prices).pct_change().rolling(window=20).std().fillna(0).values
        avg_vol = np.mean(volatility) if np.mean(volatility) > 0 else 0.01
        
        # 分段处理
        segments = max(3, len(prices) // 100)
        for i in range(segments):
            start_idx = i * len(prices) // segments
            end_idx = min((i+1) * len(prices) // segments, len(prices))
            
            if end_idx - start_idx < min_window*2:
                continue
                
            segment_prices = prices[start_idx:end_idx]
            segment_vol = np.mean(volatility[start_idx:end_idx]) if start_idx < len(volatility) else avg_vol
            
            # 根据波动率调整窗口
            window = max(min_window, int(min_window * (1 + segment_vol/avg_vol)))
            window = min(window, (end_idx - start_idx) // 4)
            
            maxima = signal.argrelextrema(segment_prices, np.greater, order=window)[0]
            minima = signal.argrelextrema(segment_prices, np.less, order=window)[0]
            
            for idx in maxima:
                abs_idx = start_idx + idx
                extrema.append((abs_idx, dates[abs_idx], prices[abs_idx], 1))
            
            for idx in minima:
                abs_idx = start_idx + idx
                extrema.append((abs_idx, dates[abs_idx], prices[abs_idx], -1))
    
    elif method == 'multi_scale':
        # 多尺度检测方法 - 使用多个窗口并合并结果
        for window in range(min_window, max_window+1, 3):
            maxima = signal.argrelextrema(prices, np.greater, order=window)[0]
            minima = signal.argrelextrema(prices, np.less, order=window)[0]
            
            for idx in maxima:
                extrema.append((idx, dates[idx], prices[idx], 1))
            
            for idx in minima:
                extrema.append((idx, dates[idx], prices[idx], -1))
    
    # 排序并过滤太近的点
    extrema.sort(key=lambda x: x[0])
    filtered_extrema = []
    min_distance = min_window // 2
    
    for point in extrema:
        if not filtered_extrema or point[0] - filtered_extrema[-1][0] >= min_distance:
            filtered_extrema.append(point)
    
    return filtered_extrema

# 步骤3: 波浪模式识别模块 - 支持多种波浪类型
def identify_wave_patterns(extrema, min_confidence=0.5):
    """
    识别极值点序列中的各种波浪模式
    
    返回:
        波浪模式字典，按类型分组
    """
    patterns = {
        'impulse': [],    # 冲动波
        'correction': [], # 修正波
        'zigzag': [],     # 之字形
        'flat': [],       # 平台型
        'triangle': [],   # 三角形
        'diagonal': [],   # 对角线
        'forming': []     # 形成中的波浪
    }
    
    # 至少需要3个点才能形成基本波浪
    if len(extrema) < 3:
        return patterns
    
    # 检查所有可能的起点
    for i in range(len(extrema) - 2):
        # 1. 识别ZigZag模式 (3点结构)
        if i+2 < len(extrema):
            points = extrema[i:i+3]
            # 检查点类型交替
            if points[0][3] != points[1][3] and points[1][3] != points[2][3]:
                # 添加简单zigzag
                confidence = 0.6
                patterns['zigzag'].append({
                    'points': points,
                    'confidence': confidence,
                    'start_idx': i,
                    'end_idx': i+2
                })
        
        # 2. 识别冲动波 (5点结构)
        if i+4 < len(extrema):
            points = extrema[i:i+5]
            # 检查点类型交替
            valid = True
            for j in range(1, 5):
                if points[j][3] == points[j-1][3]:
                    valid = False
                    break
            
            if valid:
                # 判断是上升还是下降
                is_uptrend = points[0][3] == -1  # 从谷开始是上升趋势
                
                # 验证艾略特规则
                rules_score = 0.0
                
                if is_uptrend:
                    # 浪2不应超过浪1起点
                    if points[2][2] > points[0][2]:
                        rules_score += 0.2
                    
                    # 浪3应该比浪1长
                    wave1_height = points[1][2] - points[0][2]
                    wave3_height = points[3][2] - points[2][2]
                    if wave3_height > wave1_height:
                        rules_score += 0.2
                    
                    # 浪4不应进入浪1价格区域
                    if points[4][2] > points[1][2]:
                        rules_score += 0.2
                else:
                    # 下降趋势规则
                    if points[2][2] < points[0][2]:
                        rules_score += 0.2
                    
                    wave1_height = points[0][2] - points[1][2]
                    wave3_height = points[2][2] - points[3][2]
                    if wave3_height > wave1_height:
                        rules_score += 0.2
                    
                    if points[4][2] < points[1][2]:
                        rules_score += 0.2
                
                # 计算总体置信度
                confidence = 0.4 + rules_score
                
                if confidence >= min_confidence:
                    patterns['impulse'].append({
                        'points': points,
                        'confidence': confidence,
                        'is_uptrend': is_uptrend,
                        'start_idx': i,
                        'end_idx': i+4
                    })
        
        # 3. 识别修正波 (3点结构 A-B-C)
        if i+2 < len(extrema):
            points = extrema[i:i+3]
            # 检查点类型交替
            if points[0][3] != points[1][3] and points[1][3] != points[2][3]:
                # 验证ABC修正波的特征
                is_uptrend = points[0][3] == 1  # 从峰开始是下降趋势的修正
                
                # 简单验证: B点应该回撤A-C距离的一部分
                a_c_height = abs(points[2][2] - points[0][2])
                a_b_height = abs(points[1][2] - points[0][2])
                b_c_height = abs(points[2][2] - points[1][2])
                
                retrace_ratio = a_b_height / a_c_height if a_c_height > 0 else 0
                
                # 给出置信度
                confidence = 0.5
                if 0.3 <= retrace_ratio <= 0.7:
                    confidence += 0.2
                
                if confidence >= min_confidence:
                    patterns['correction'].append({
                        'points': points,
                        'confidence': confidence,
                        'is_uptrend': is_uptrend,
                        'start_idx': i,
                        'end_idx': i+2
                    })
    
    # 4. 识别形成中的波浪 (使用最近的极值点)
    if len(extrema) >= 2:
        last_points = extrema[-3:] if len(extrema) >= 3 else extrema[-2:]
        
        # 检查点类型交替
        valid = True
        for j in range(1, len(last_points)):
            if last_points[j][3] == last_points[j-1][3]:
                valid = False
                break
        
        if valid:
            is_uptrend = last_points[-1][3] == -1  # 最后一点是谷意味着下一步可能上升
            
            patterns['forming'].append({
                'points': last_points,
                'confidence': 0.7,  # 形成中的波浪置信度较高
                'is_uptrend': is_uptrend,
                'start_idx': len(extrema) - len(last_points),
                'end_idx': len(extrema) - 1
            })
    
    return patterns

# 步骤4: 波浪状态分析模块
def analyze_wave_status(patterns, prices, extrema, current_date):
    """
    分析当前波浪状态并预测可能的走势
    
    参数:
        patterns: 识别出的各类波浪模式
        prices: 价格序列
        extrema: 极值点列表
        current_date: 当前日期
    
    返回:
        波浪状态分析结果
    """
    # 获取各类型波浪
    impulse_waves = patterns['impulse']
    correction_waves = patterns['correction']
    zigzag_waves = patterns['zigzag']
    forming_waves = patterns['forming']
    
    current_price = prices[-1]
    
    # 如果没有识别出任何波浪模式
    if not any([impulse_waves, correction_waves, zigzag_waves, forming_waves]):
        # 简单基于最近极值点进行分析
        if len(extrema) >= 2:
            recent_points = extrema[-3:] if len(extrema) >= 3 else extrema
            last_point = recent_points[-1]
            
            # 计算最后一个极值点到现在的时间和价格变化
            days_since = (current_date - last_point[1]).days
            price_rel = (current_price - last_point[2]) / last_point[2]
            
            # 判断趋势方向
            is_uptrend = last_point[3] == -1  # 最后极值点是谷，当前趋势应为上升
            
            # 计算趋势强度
            trend_strength = min(1.0, abs(price_rel) * 10)  # 0-1之间的值
            
            # 判断可能的波浪阶段
            wave_stage = "unknown"
            if len(recent_points) >= 3:
                if recent_points[0][3] == -1 and recent_points[1][3] == 1 and recent_points[2][3] == -1:
                    wave_stage = "impulse_start" if price_rel > 0 else "correction"
                elif recent_points[0][3] == 1 and recent_points[1][3] == -1 and recent_points[2][3] == 1:
                    wave_stage = "correction_start" if price_rel < 0 else "impulse"
            else:
                wave_stage = "consolidation"
            
            return {
                "status": "partial_pattern",
                "message": "基于极值点的简单分析",
                "trend": "upward" if is_uptrend else "downward",
                "trend_strength": trend_strength,
                "recent_extrema_count": len(recent_points),
                "days_since_last_extrema": days_since,
                "price_change_since_last": price_rel * 100,  # 转为百分比
                "possible_stage": wave_stage,
                "confidence": 0.4 + trend_strength * 0.3,  # 基于趋势强度调整置信度
                "prediction": {
                    "next_move": "impulse_up" if last_point[3] == -1 else "correction",
                    "message": f"可能正在形成{'上升' if is_uptrend else '下降'}趋势的新波段",
                    "target_levels": [current_price * (1.05 if is_uptrend else 0.95)],
                    "primary_target": current_price * (1.05 if is_uptrend else 0.95),
                    "probability": 0.5
                }
            }
        
        return {
            "status": "insufficient_data",
            "message": "数据不足，无法进行波浪分析",
            "suggestion": "考虑使用更长的历史数据或调整极值点检测参数"
        }
    
    # 检查是否有正在形成中的波浪
    if forming_waves:
        best_forming = max(forming_waves, key=lambda w: w['confidence'])
        return _analyze_forming_wave(best_forming, extrema, current_price, current_date)
    
    # 检查冲动波
    if impulse_waves:
        # 按置信度排序
        impulse_waves.sort(key=lambda w: w['confidence'], reverse=True)
        best_impulse = impulse_waves[0]
        
        # 检查冲动波是否完整
        if best_impulse['end_idx'] == len(extrema) - 1:
            points = best_impulse['points']
            is_uptrend = best_impulse['is_uptrend']
            
            # 计算可能的目标位
            targets = []
            start_price = points[0][2]
            end_price = points[-1][2]
            
            # 确定当前位置相对于极值点的情况
            price_change = (current_price - end_price) / end_price
            
            # 添加斐波那契目标位
            wave_height = abs(end_price - start_price)
            for level in [0.382, 0.5, 0.618]:
                retracement = end_price - wave_height * level if is_uptrend else end_price + wave_height * level
                targets.append(retracement)
            
            # 预测结果
            message = f"已完成{'上升' if is_uptrend else '下降'}冲动波结构，预计进入{'下跌' if is_uptrend else '上涨'}修正波"
            
            return {
                'status': 'impulse_complete',
                'message': message,
                'confidence': best_impulse['confidence'],
                'prediction': {
                    'next_move': 'correction',
                    'message': message,
                    'target_levels': targets,
                    'primary_target': targets[1],  # 0.5回撤位
                    'probability': best_impulse['confidence'] * 0.7
                }
            }
        else:
            # 冲动波正在进行中
            return _analyze_forming_wave(best_impulse, extrema, current_price, current_date)
                
    # 检查修正波
    if correction_waves:
        correction_waves.sort(key=lambda w: w['confidence'], reverse=True)
        correction = correction_waves[0]
        
        # 检查修正波是否完整
        if correction['end_idx'] == len(extrema) - 1:
            points = correction['points']
            is_uptrend = not correction['is_uptrend']  # 修正波方向与主趋势相反
            
            # 计算目标位
            targets = []
            start_price = points[0][2]
            end_price = points[-1][2]
            
            # 添加斐波那契扩展目标位
            wave_height = abs(end_price - start_price)
            for level in [1.0, 1.618, 2.0]:
                extension = end_price + wave_height * level if is_uptrend else end_price - wave_height * level
                targets.append(extension)
            
            return {
                'status': 'correction_complete',
                'message': f'已完成{"下跌" if is_uptrend else "上涨"}修正波',
                'confidence': correction['confidence'],
                'prediction': {
                    'next_move': '新冲动波',
                    'message': f'可能开始新的{"上升" if is_uptrend else "下降"}冲动波',
                    'target_levels': targets,
                    'primary_target': targets[1] if len(targets) > 1 else targets[0],
                    'probability': correction['confidence'] * 0.6
                }
            }
    
    # 默认返回基本分析
    return {
        'status': 'basic_analysis',
        'message': '无法确定明确的波浪模式',
        'suggestion': '考虑使用其他技术指标辅助分析'
    }

# 辅助函数 - 分析形成中的波浪
def _analyze_forming_wave(wave, extrema, current_price, current_date):
    """分析正在形成中的波浪"""
    points = wave['points']
    last_point = points[-1]
    last_idx = wave['end_idx']
    
    # 计算时间和价格变化
    days_since = (current_date - last_point[1]).astype('timedelta64[D]').astype(int)
    price_rel = (current_price - last_point[2]) / last_point[2]
    
    # 判断趋势方向
    is_uptrend = wave['is_uptrend']
    
    # 当前形成的可能是新的极值点
    forming_new_point = abs(price_rel) > 0.05 and days_since > 5
    
    # 下一个点的预期类型
    next_point_type = -1 if last_point[3] == 1 else 1
    
    # 确定当前波浪阶段
    current_wave_count = len(points)
    next_wave = current_wave_count + 1
    
    # 计算可能的目标位
    targets = []
    
    # 根据不同的波浪阶段计算目标
    if current_wave_count == 1:  # 只有一个点
        # 简单估计：5%的价格变动
        targets = [current_price * (1.05 if next_point_type == 1 else 0.95)]
    
    elif current_wave_count >= 2:
        # 使用前两个点的价格差异来估计目标
        wave1_height = abs(points[1][2] - points[0][2])
        
        if next_wave == 3:  # 预测浪3
            # 浪3通常是最强的，可能是浪1的1.618或2.618倍
            for level in [1.618, 2.0, 2.618]:
                if is_uptrend:
                    target = points[1][2] + wave1_height * level
                else:
                    target = points[1][2] - wave1_height * level
                targets.append(target)
                
        elif next_wave == 4:  # 预测浪4
            # 浪4通常回撤浪3的38.2%或50%
            if current_wave_count >= 3:
                wave3_height = abs(points[2][2] - points[1][2])
                for level in [0.236, 0.382, 0.5]:
                    if is_uptrend:
                        target = points[2][2] - wave3_height * level
                    else:
                        target = points[2][2] + wave3_height * level
                    targets.append(target)
            else:
                targets = [current_price * (0.95 if is_uptrend else 1.05)]
                
        elif next_wave == 5:  # 预测浪5
            # 浪5通常等于浪1或者是浪1的0.618倍或1.618倍
            for level in [0.618, 1.0, 1.618]:
                if is_uptrend:
                    target = points[3][2] + wave1_height * level
                else:
                    target = points[3][2] - wave1_height * level
                targets.append(target)
        else:
            # 超过5个点，可能是复杂结构
            targets = [current_price * (1.03 if is_uptrend else 0.97)]
    
    if not targets:  # 如果无法确定目标，返回简单估计
        targets = [current_price * (1.05 if next_point_type == 1 else 0.95)]
        
    # 组织返回信息
    primary_target = targets[1] if len(targets) >= 2 else targets[0]
    
    return {
        "status": "forming_wave",
        "message": f"正在形成第{next_wave}浪",
        "wave_count": current_wave_count,
        "trend": "upward" if is_uptrend else "downward",
        "current_stage": current_wave_count,
        "next_stage": next_wave,
        "forming_new_point": forming_new_point,
        "days_since_last_point": days_since,
        "price_change_since_last": price_rel * 100,  # 转为百分比
        "confidence": wave['confidence'],
        "prediction": {
            "next_point_type": "peak" if next_point_type == 1 else "trough",
            "next_wave": next_wave,
            "target_levels": targets,
            "primary_target": primary_target,
            "message": f"当前处于第{current_wave_count}浪，正在形成第{next_wave}浪",
            "probability": wave['confidence'] * (0.7 if forming_new_point else 0.5)
        }
    }

# 步骤5: 集成分析函数 - 修复版
def analyze_elliott_waves(symbol, start_date="2023-01-01", end_date=None):
    """集成分析函数 - 便于单独测试"""
    # 1. 获取数据
    df = get_stock_data(symbol, start_date, end_date)
    if df.empty:
        return {'error': f'无法获取{symbol}数据'}
    
    # 2. 检测极值点
    prices = df['close'].values
    dates = df.index.values
    extrema = detect_extrema(prices, dates, method='adaptive', window_range=(5, 15))
    
    print(f"--{symbol}-- 检测到 {len(extrema)} 个极值点")
    
    # 3. 识别波浪模式
    patterns = identify_wave_patterns(extrema, min_confidence=0.5)
    
    pattern_counts = {k: len(v) for k, v in patterns.items()}
    print(f"识别出的波浪模式: {pattern_counts}")
    
    # 4. 分析当前状态 - 修复此处，传入extrema参数
    status = analyze_wave_status(patterns, prices, extrema, dates[-1])
    
    # 5. 显示结果
    print(f"\n=== {symbol} 艾略特波浪分析 ===")
    print(f"分析状态: {status['status']}")
    print(f"分析描述: {status['message']}")
    
    if 'prediction' in status:
        pred = status['prediction']
        print(f"\n== 趋势预测 ==")
        print(f"{pred['message']}")
        
        if 'target_levels' in pred:
            print(f"目标价位: {[round(t, 2) for t in pred['target_levels']]}")
            print(f"主要目标价位: {round(pred['primary_target'], 2)}")
        
        if 'probability' in pred:
            print(f"预测置信度: {pred['probability']:.2f}")
    
    return {
        'symbol': symbol,
        'extrema_count': len(extrema),
        'patterns': pattern_counts,
        'status': status
    }

# 测试函数 - 只对单一模块进行测试
def test_extrema_detection(symbol="AAPL", start_date="2023-01-01"):
    """测试极值点检测"""
    df = get_stock_data(symbol, start_date)
    prices = df['close'].values
    dates = df.index.values
    
    # 测试不同方法
    methods = ['simple', 'adaptive', 'multi_scale']
    for method in methods:
        extrema = detect_extrema(prices, dates, method=method)
        print(f"--{symbol}-- 方法 {method}: 检测到 {len(extrema)} 个极值点")
        
        # 可视化
        plt.figure(figsize=(15, 7))
        plt.plot(dates, prices, label='价格')
        
        # 标记极值点
        peaks_x = [dates[p[0]] for p in extrema if p[3] == 1]
        peaks_y = [p[2] for p in extrema if p[3] == 1]
        
        troughs_x = [dates[p[0]] for p in extrema if p[3] == -1]
        troughs_y = [p[2] for p in extrema if p[3] == -1]
        
        plt.scatter(peaks_x, peaks_y, color='red', label='峰值')
        plt.scatter(troughs_x, troughs_y, color='green', label='谷值')
        
        plt.title(f"{symbol} 极值点检测 - {method}方法")
        plt.legend()
        plt.show()

# 主函数
if __name__ == "__main__":
    
    # 1. 先测试单一股票的极值点检测
    test_extrema_detection("AAPL", "2023-01-01")
    #return

    # 逐步测试各个模块
    symbols = ["AAPL", "MSFT", "TSLA", "PLTR", "NVDA"]
    # 2. 再测试完整分析流程
    for symbol in symbols:
        result = analyze_elliott_waves(symbol, "2023-01-01")
        print("="*50 + "\n")
