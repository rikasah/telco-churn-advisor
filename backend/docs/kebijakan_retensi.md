# Kebijakan Retensi Pelanggan

## Siapa yang dianggap "berisiko tinggi" untuk churn?
Pelanggan diklasifikasikan berisiko tinggi (high risk) jika model prediksi churn kami menghasilkan probabilitas churn di atas 50%. Faktor-faktor yang paling sering muncul pada pelanggan berisiko tinggi: kontrak month-to-month, tenure (lama berlangganan) di bawah 12 bulan, tidak memiliki add-on Tech Support/Online Security, dan metode pembayaran electronic check.

## Penawaran retensi apa yang tersedia untuk pelanggan berisiko tinggi?
1. Diskon 15% selama 3 bulan jika bersedia upgrade ke kontrak 1 tahun.
2. Trial gratis Tech Support + Online Security selama 2 bulan.
3. Cashback untuk migrasi metode pembayaran ke bank transfer otomatis.
4. Konsultasi gratis dengan tim customer success untuk meninjau ulang paket layanan yang sesuai kebutuhan.

## Bagaimana tim customer service menggunakan skor risiko churn?
Skor risiko (churn_probability) dari model prediksi digunakan untuk memprioritaskan follow-up: pelanggan dengan risiko "high" dihubungi proaktif dalam 3 hari kerja, risiko "medium" dikirim email retensi otomatis, dan risiko "low" tidak memerlukan tindakan khusus.

## Apakah penawaran retensi otomatis diberikan?
Tidak. Semua penawaran retensi memerlukan konfirmasi dan persetujuan dari pelanggan itu sendiri melalui kanal resmi (aplikasi, call center, atau email resmi perusahaan). Tim tidak pernah meminta pelanggan mengirimkan data kartu kredit lengkap atau password akun melalui chat.
