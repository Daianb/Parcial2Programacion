from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select
from tabulate import tabulate

class Estudiantes(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Nombre: str
    Apodo: str
    Edad: Optional[int] = None

estudiante_1 = Estudiantes(Nombre="Sebastian", Apodo="Sebas", Edad=22)
estudiante_2 = Estudiantes(Nombre="Viviana", Apodo="Karol", Edad= 20)
estudiante_3 = Estudiantes(Nombre="Camilo", Apodo="Milo", Edad=19)

class Semestre(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Semestrequeva: str
    Ciudad: str

semestre_1 = Semestre(Semestrequeva="Quinto", Ciudad="Quimbaya")
semestre_2 = Semestre(Semestrequeva="Quinto", Ciudad="Circasia")
semestre_3 = Semestre(Semestrequeva="Quinto", Ciudad="Quimbaya")

engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)
with Session(engine) as session:
    session.add(estudiante_1)
    session.add(estudiante_2)
    session.add(estudiante_3)
    #session.commit()
    session.add(semestre_1)
    session.add(semestre_2)
    session.add(semestre_3)
    session.commit()

    with Session(engine) as session:
        statement = select(Estudiantes).where(Estudiantes.Nombre ==  "Viviana")
        result = session.exec(statement).first()
        print(f"| {'id':^3} | {'Nombre':^20} | {'Apodo':^20} | {'Edad':^5} |")
        print(f"|{'-'*5}|{'-'*22}|{'-'*22}|{'-'*7}|")
        print(f"| {result.id:^3} | {result.Nombre:^20} | {result.Apodo:^20} | {result.Edad:^5} |")