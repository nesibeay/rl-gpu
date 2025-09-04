import csv, os

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["update","global_step","mean_return","mean_ep_len","fps","eval_return"])

    def write(self, update:int, global_step:int, mean_ret:float, mean_len:float, fps:float, eval_return=None):
        self.w.writerow([
            update, global_step,
            f"{mean_ret:.3f}", f"{mean_len:.3f}", f"{fps:.1f}",
            "" if eval_return is None else f"{eval_return:.3f}"
        ])
        self.f.flush()

    def close(self):
        try: self.f.close()
        except: pass
