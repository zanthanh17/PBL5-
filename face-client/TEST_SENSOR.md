# 🧪 Test Sensor HC-SR04 + LED

## Quick Test Commands

### 1. Test với interval 5 giây (mặc định)
```bash
python3 src/sensor_controller.py
# Hoặc
python3 src/sensor_controller.py 5
```

**Output:**
```
Check interval: 5.0s
Monitoring started (checking every 5.0s)...
>>> [10:01:35] Person detected at 93.77cm - LED ON
>>> [10:01:45] Person left - LED OFF
```

### 2. Test với interval tùy chỉnh
```bash
# Test mỗi 3 giây
python3 src/sensor_controller.py 3

# Test mỗi 10 giây
python3 src/sensor_controller.py 10

# Test real-time (0.2 giây)
python3 src/sensor_controller.py 0.2
```

### 3. Test riêng từng component

**Test chỉ HC-SR04 sensor:**
```bash
python3 src/sensor_controller.py sensor
```

**Test chỉ LED:**
```bash
python3 src/sensor_controller.py led
```

## Config trong Client

Edit `config/client.yaml`:

```yaml
sensor:
  enabled: true  # Bật sensor
  check_interval: 5.0  # Đo mỗi 5 giây (cho test)
```

### Check Interval Recommendations:

| Interval | Use Case | CPU Impact |
|----------|----------|------------|
| 0.2s | Production (real-time) | ~3% |
| 1.0s | Balanced | ~1% |
| 5.0s | Testing/Debug | <1% |
| 10.0s | Very slow monitoring | <1% |

## Debug Mode

Enable debug logs để thấy distance readings:

```python
# Trong sensor_controller.py hoặc trước khi chạy:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Output:**
```
[DEBUG] Distance: 93.77cm
[DEBUG] Distance: 85.23cm
[INFO] Person detected at 85.23cm
```

## Test Scenarios

### Scenario 1: Verify Detection
```bash
python3 src/sensor_controller.py 2

# Di chuyển tay trước sensor (< 100cm)
# Expected: "Person detected" + LED ON
```

### Scenario 2: Verify Auto-off
```bash
python3 src/sensor_controller.py 1

# Di chuyển tay ra xa (> 100cm)
# Đợi 10 giây
# Expected: "Person left" + LED OFF
```

### Scenario 3: Multiple Triggers
```bash
python3 src/sensor_controller.py 0.5

# Di chuyển tay vào/ra nhiều lần
# LED nên bật/tắt theo
```

## Expected Behavior

### Normal Operation:
1. **Person approaches** (distance < 100cm):
   - Log: "Person detected at Xcm"
   - LED: ON ⚡
   - TTS: "Xin chào" (if TTS enabled)

2. **Person stays** (distance < 100cm):
   - LED stays ON
   - Timer resets

3. **Person leaves** (distance > 100cm):
   - Wait `led_on_duration` (default 10s)
   - Log: "Person left"
   - LED: OFF 💡

### Check Interval Effect:

**Fast interval (0.2s)**:
- ✅ Real-time response
- ✅ Quick detection
- ⚠️ More CPU usage
- ⚠️ More logs

**Slow interval (5s)**:
- ✅ Less CPU usage
- ✅ Less logs
- ✅ Good for testing
- ⚠️ Delayed response

## Troubleshooting

### Issue: "Person detected" fires immediately

**Cause**: Có vật cản < 100cm

**Fix**:
```yaml
# Giảm trigger distance
sensor:
  trigger_distance: 50.0  # Chỉ khi rất gần
```

### Issue: Too many logs

**Cause**: Check interval quá nhanh

**Fix**:
```bash
# Test với interval chậm hơn
python3 src/sensor_controller.py 5
```

### Issue: LED không tắt

**Cause**: Vẫn có vật < trigger_distance

**Fix**:
- Di chuyển vật ra xa
- Hoặc tăng trigger_distance

## Performance Test

Test CPU usage với các interval khác nhau:

```bash
# Terminal 1: Run sensor
python3 src/sensor_controller.py 0.2

# Terminal 2: Monitor CPU
top -p $(pgrep -f sensor_controller)
```

| Interval | CPU % | Memory | Recommendation |
|----------|-------|--------|----------------|
| 0.2s | ~3% | 15MB | ✅ Production |
| 1.0s | ~1% | 15MB | ✅ Balanced |
| 5.0s | <1% | 15MB | ✅ Testing |

## Integration Test

Test sensor với full client:

```bash
# 1. Enable sensor
nano config/client.yaml
# sensor: enabled: true, check_interval: 5.0

# 2. Run client
python3 src/client.py

# 3. Di chuyển tay trước sensor
# Expected:
#   - "Person detected" log
#   - LED ON
#   - TTS "Xin chào"
#   - Camera recognition starts
```

## Tips

1. **Test từng bước**:
   ```bash
   # Step 1: Test sensor only
   python3 src/sensor_controller.py sensor
   
   # Step 2: Test LED only
   python3 src/sensor_controller.py led
   
   # Step 3: Test full controller
   python3 src/sensor_controller.py 5
   ```

2. **Use slow interval for debugging**:
   ```bash
   # Easy to see what's happening
   python3 src/sensor_controller.py 10
   ```

3. **Check logs with timestamp**:
   ```bash
   python3 src/sensor_controller.py 2 | tee sensor_test.log
   ```

4. **Test in production config**:
   ```bash
   # Use config from client.yaml
   python3 src/client.py
   # (with sensor: enabled: true)
   ```

## Quick Start for Testing

```bash
# Clone repo
cd ~/PBL5-/face-client

# Test với 5 giây interval
python3 src/sensor_controller.py 5

# Di chuyển tay trước sensor
# Xem log: "Person detected at Xcm"
# LED nên sáng

# Ctrl+C để stop
```

---

**Note**: Sau khi test OK, set `check_interval: 0.2` trong config cho production (real-time response).

