from dataclasses import dataclass

@dataclass
class Ubot:
    id: int
    ang: float
    dist: float
    Out: int
    comm_ok: bool = True  # Estado de la comunicación con el robot