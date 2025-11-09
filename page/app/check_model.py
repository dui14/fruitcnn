import tensorflow as tf

# Load model
model = tf.keras.models.load_model('fruit_model_full.h5')

# In thông tin model
print("="*50)
print("THÔNG TIN MODEL")
print("="*50)

# In summary
model.summary()

print("\n" + "="*50)
print("INPUT SHAPE")
print("="*50)
print(f"Input shape: {model.input_shape}")
print(f"Expected input: {model.input_shape[1:]} (bỏ qua batch size)")

print("\n" + "="*50)
print("OUTPUT SHAPE")
print("="*50)
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")

# Tính toán kích thước ảnh cần thiết
input_shape = model.input_shape[1:]
if len(input_shape) == 3:
    height, width, channels = input_shape
    print("\n" + "="*50)
    print("KÍCH THƯỚC ẢNH CẦN THIẾT")
    print("="*50)
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Total features: {height * width * channels}")
elif len(input_shape) == 1:
    print("\n" + "="*50)
    print("MODEL DẠNG FLATTEN")
    print("="*50)
    print(f"Expected flatten size: {input_shape[0]}")
    
    # Thử tính ngược
    # Nếu 12800 và có 2 channels
    if input_shape[0] == 12800:
        size_per_channel = 12800 / 2
        if size_per_channel == 6400:
            side = int(6400 ** 0.5)
            print(f"Có thể là: {side}x{side}x2 = {side * side * 2}")
    
    # Nếu 86528
    if input_shape[0] == 86528:
        # Thử với 3 channels
        size_per_channel = 86528 / 3
        side = int((size_per_channel) ** 0.5)
        print(f"Nếu 3 channels: {side}x{side}x3 = {side * side * 3}")
        
        # Thử với 2 channels
        size_per_channel = 86528 / 2
        side = int((size_per_channel) ** 0.5)
        print(f"Nếu 2 channels: {side}x{side}x2 = {side * side * 2}")
        
        # Thử với 1 channel
        side = int((86528) ** 0.5)
        print(f"Nếu 1 channel: {side}x{side}x1 = {side * side * 1}")