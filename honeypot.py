import logging
import socket
import threading
import time
import random


def create_honeypots():
    logging.info("Creating honeypots...")
    tcp_honeypot = create_tcp_honeypot()
    http_honeypot = create_http_honeypot()
    return [tcp_honeypot, http_honeypot]


def start_honeypots(honeypots):
    logging.info("Starting honeypots...")
    for honeypot in honeypots:
        honeypot.start()


def create_tcp_honeypot():
    logging.info("Creating TCP honeypot...")
    tcp_honeypot = TCPhoneypot()
    return tcp_honeypot


def create_http_honeypot():
    logging.info("Creating HTTP honeypot...")
    http_honeypot = HTTPhoneypot()
    return http_honeypot


class TCPhoneypot:
    def __init__(self):
        self.host = "0.0.0.0"  # Listening on all interfaces
        self.port = 12345  # Example port

    def start(self):
        logging.info("TCP honeypot started.")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                while True:
                    client_socket, address = server_socket.accept()
                    logging.info(f"Incoming TCP connection from: {address[0]}")
                    threading.Thread(target=self.handle_connection, args=(client_socket, address)).start()
        except Exception as e:
            logging.error(f"Error in TCP honeypot: {str(e)}")

    def handle_connection(self, client_socket, address):
        try:
            time.sleep(random.uniform(0.5, 3.0))
            logging.info(f"Attacker executed a command from IP {address[0]}...")
            time.sleep(random.uniform(0.1, 1.0))
            logging.info(f"Attacker accessed a vulnerable file from IP {address[0]}...")

            # Simulate a vulnerability
            if random.random() < 0.2:
                logging.info(f"Vulnerability exploited by the attacker from IP {address[0]}!")

            # Simulate additional actions
            if random.random() < 0.1:
                logging.info(f"Attacker attempted a privilege escalation from IP {address[0]}...")
            if random.random() < 0.05:
                logging.info(f"Attacker performed reconnaissance from IP {address[0]}...")
        except Exception as e:
            logging.error(f"Error handling TCP connection: {str(e)}")
        finally:
            client_socket.close()


class HTTPhoneypot:
    def __init__(self):
        self.host = "0.0.0.0"  # Listening on all interfaces
        self.port = 80  # Example port

    def start(self):
        logging.info("HTTP honeypot started.")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                while True:
                    client_socket, address = server_socket.accept()
                    logging.info(f"Incoming HTTP request from: {address[0]}")
                    threading.Thread(target=self.handle_request, args=(client_socket, address)).start()
        except Exception as e:
            logging.error(f"Error in HTTP honeypot: {str(e)}")

    def handle_request(self, client_socket, address):
        try:
            time.sleep(random.uniform(0.5, 3.0))
            logging.info(f"Attacker sent a malicious payload from IP {address[0]}...")
            time.sleep(random.uniform(0.1, 1.0))
            logging.info(f"Attacker attempted a SQL injection from IP {address[0]}...")

            # Simulate a vulnerability
            if random.random() < 0.1:
                logging.info(f"Vulnerability exploited by the attacker from IP {address[0]}!")

            # Simulate additional actions
            if random.random() < 0.05:
                logging.info(f"Attacker performed XSS attack from IP {address[0]}...")
            if random.random() < 0.03:
                logging.info(f"Attacker attempted a CSRF attack from IP {address[0]}...")
        except Exception as e:
            logging.error(f"Error handling HTTP request: {str(e)}")
        finally:
            client_socket.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    honeypots = create_honeypots()
    start_honeypots(honeypots)
