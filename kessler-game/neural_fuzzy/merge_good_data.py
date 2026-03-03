
import pandas as pd
from pathlib import Path

base = Path("good_data")
out  = Path("data")
out.mkdir(parents=True, exist_ok=True)

m_files = [base/"1_maneuver.csv", base/"2_maneuver.csv", base/"3_maneuver.csv"]
c_files = [base/"1_combat.csv",   base/"2_combat.csv",   base/"3_combat.csv"]

man = pd.concat([pd.read_csv(f) for f in m_files], ignore_index=True)
com = pd.concat([pd.read_csv(f) for f in c_files], ignore_index=True)

man.to_csv(out/"maneuver.csv", index=False)
com.to_csv(out/"combat.csv", index=False)

print("wrote data/maneuver.csv rows=", len(man))
print("wrote data/combat.csv   rows=", len(com))