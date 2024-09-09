import cv2
import numpy as np

# Görüntüyü yükle
image_path = '2adet.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Görüntü yüklenemedi.")
else:
    # Gri tonlamaya çevir
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian bulanıklık, kernel boyutu daha küçük
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Otsu ile adaptif eşikleme
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfolojik işlemler (daha az tekrar)
    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Konturları bul ve filtrele
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 30  # Küçük alanlar için daha düşük eşik
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    # Konturları çiz
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 1)

    # Tespit edilen toplam tartar sayısı
    cockroach_count = len(filtered_contours)

    # Sağ üst köşeye metni ekle
    text = f"{cockroach_count} adet canli"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, 1, 2)
    text_x = output_image.shape[1] - text_size[0] - 10  # Sağ üst köşe
    text_y = 30  # Yukarıdan 30 piksel boşluk
    cv2.putText(output_image, text, (text_x, text_y), font, 1, (0, 0, 255), 2)  # Kırmızı renk ve kalınlık

    # Sonuçları yazdır
    print(f'Toplam tespit edilen tartar sayısı: {cockroach_count}')

    # İşlenmiş görüntüleri kaydet
    cv2.imwrite('output_image_with_count.png', output_image)

    # İsteğe bağlı olarak görüntü göster
    cv2.imshow('Tartar Tespiti', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
