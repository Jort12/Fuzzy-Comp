
from kesslergame import KesslerGraphics
from kesslergame.graphics.graphics_tk import GraphicsTK
from kesslergame.graphics.graphics_ue import GraphicsUE

class GraphicsBoth(KesslerGraphics):
    def __init__(self):
        self.ue = GraphicsUE()
        self.tk = GraphicsTK({})

    def start(self, scenario):
        self.ue.start(scenario)
        self.tk.start(scenario)

    def update(self, score, ships, asteroids, bullets, mines):
        self.ue.update(score, ships, asteroids, bullets, mines)
        self.tk.update(score, ships, asteroids, bullets, mines)

    def close(self):
        self.ue.close()
        self.tk.close()
