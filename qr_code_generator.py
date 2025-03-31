import qrcode

# Ton URL
url = "https://ml-csc1117.streamlit.app/"

# Génération du QR Code
qr = qrcode.make(url)

# Sauvegarde en image
qr.save("qr_f1_simulator.png")
