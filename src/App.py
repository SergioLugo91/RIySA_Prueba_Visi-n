import time
import cv2
from flask import Flask, render_template, Response, redirect, url_for

class Interface:
    """Clase para manejar la interfaz web Flask del sistema de robots."""
    
    def __init__(self, robot_comm):
        """
        Inicializa la interfaz web.
        
        Args:
            robot_comm: Instancia de RobotComm para acceder a estados y log
        """
        self.robot_comm = robot_comm
        self.current_frame = None
        
        # Flask
        self.app = Flask(__name__)
        self._setup_routes()
    
    def update_frame(self, frame):
        """
        Actualiza el frame actual para el streaming.
        
        Args:
            frame: Frame de OpenCV (numpy array) a mostrar
        """
        self.current_frame = frame
    
    def gen_frames(self):
        """Generador de frames para el streaming de video."""
        while True:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def _setup_routes(self):
        """Configura las rutas de Flask."""
        @self.app.route('/')
        def index():
            return render_template("index.html", 
                                   states=self.robot_comm.robot_states,
                                   comm_status=self.robot_comm.robot_comm_status)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.gen_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/start')
        def start_fight():
            for rid in self.robot_comm.robot_states:
                self.robot_comm.robot_states[rid] = "peleando"
            self.robot_comm.log("PELEA →", "Se inició la pelea")
            return redirect(url_for('index'))

        @self.app.route('/stop')
        def stop_fight():
            for rid in self.robot_comm.robot_states:
                self.robot_comm.robot_states[rid] = "fuera de combate"
            self.robot_comm.log("PELEA →", "Se detuvo la pelea")
            return redirect(url_for('index'))
    
    def run_server(self, host="0.0.0.0", port=5000, debug=True):
        """
        Inicia el servidor Flask.
        
        Args:
            host: Host donde correrá el servidor
            port: Puerto donde correrá el servidor
            debug: Modo debug de Flask
        """
        self.app.run(host=host, port=port, debug=debug)


# ---------------------
# Dummy robot_comm
# ---------------------
class DummyComm:
    def __init__(self):
        self.robot_states = {"r1": "idle", "r2": "idle"}
        self.robot_comm_status = "OK"

    def log(self, title, msg):
        print(title, msg)


if __name__ == "__main__":
    interface = Interface(DummyComm())
    interface.run_server(debug=True)