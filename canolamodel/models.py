from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Fertilizer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    N = db.Column(db.Float, nullable=False)
    P2O5 = db.Column(db.Float, nullable=False)
    K2O = db.Column(db.Float, nullable=False)
    Ca = db.Column(db.Float, nullable=False)
    Mg = db.Column(db.Float, nullable=False)
    S = db.Column(db.Float, nullable=False)
    cost = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<Fertilizer {self.name}>'
