import socket

def create_wifi_sender(ip="255.255.255.255", port=8888):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", port))
    return sock

def send_data(sock, message, ip, port):
    sock.sendto(message.encode(), (ip, port))
    print(f"[Enviado -> {ip}:{port}] {message}")