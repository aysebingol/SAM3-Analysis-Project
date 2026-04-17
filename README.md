SAM 3 ile Yaprak Segmentasyonu ve Performans Analizi

Bu proje, otonom tarım araçları için SAM 3 (Segment Anything Model) kullanarak yaprak segmentasyonu (bölütleme) ve modelin başarısını ölçmek için geliştirilen analiz araçlarını içermektedir.
 Proje Kapsamı
Ozan Yılmaz tarafından belirtilen gereksinimler doğrultusunda:
•	Instance Segmentation: Her bir yaprak bağımsız bir nesne olarak ele alınmıştır.
•	Polygon Etiketleme: Roboflow üzerinde poligon yöntemiyle hassas etiketleme yapılmıştır.
•	SAM 3 Entegrasyonu: Hazırlanan veri seti SAM 3 model formatına dönüştürülerek test edilmiştir.
 Performans Metrikleri ve Doğrulama (Validation)
Projenin en kritik aşaması, modelin başarısını sadece görsel olarak değil, piksel düzeyinde bilimsel metriklerle ölçmektir.
<img width="464" height="213" alt="image" src="https://github.com/user-attachments/assets/acb9c7bb-709c-4e36-bcae-da7e20db750b" />

 Gelişmiş Doğrulama Metodolojisi
Hatalı ölçümleri (ilk etapta çıkan 1.0 skoru) engellemek için şu "Ground Truth" boru hattı geliştirilmiştir:
1.	Roboflow'dan gelen COCO JSON koordinatları, özel bir Python scripti ile Binary Mask (Siyah-Beyaz) görsellere dönüştürülmüştür.
2.	SAM 3 çıktıları, bu gerçek maskelerle piksel bazlı karşılaştırılarak gerçek IoU değerleri elde edilmiştir.
 Dosya Yapısı
•	eval_metrics.py: IoU, Precision ve Recall hesaplayan analiz scripti.
•	json_to_mask.py: JSON koordinatlarını siyah-beyaz maskeye dönüştüren araç.
•	gercek_performans_raporu.csv: Tüm test görsellerinin detaylı başarı dökümü.
 Gelecek Çalışmalar
•	Threshold Optimization: 0.5 olan eşik değerinin yükseltilerek segmentasyon sınırlarının daraltılması.
•	Post-Processing: OpenCV (Erosion/Dilation) yöntemleriyle maske kenarlarının temizlenmesi.
