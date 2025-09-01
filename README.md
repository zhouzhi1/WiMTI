# WiMTI
WiMTI is a multi-task neural network designed for both **identity recognition** and **posture classification** from WiFi Channel State Information (CSI).  
It combines dynamic/static cross-stitch units with attention mechanisms and supports multiple backbone architectures.

---

# 📦 WiMTI Public Subset – CSI-based Identity and Posture Dataset

**Download:** [https://pan.baidu.com/s/1-m_z2EHA4TOxnbG5gysMPQ?pwd=prh2](https://pan.baidu.com/s/1-m_z2EHA4TOxnbG5gysMPQ?pwd=prh2)  
**Access Code:** `prh2`

---

## 📖 Overview

This dataset is a publicly available subset derived from the proprietary CSI dataset used in the WiMTI study.  
It is released to promote reproducibility, enable peer evaluation, and encourage further research on multitask learning for WiFi sensing.

Each sample contains a CSI sequence captured with an Intel 5300 NIC and annotated with two labels:  
- **Identity** (person-level)
- **Posture** (body posture type)

---

## 📂 Dataset Structure

- **Format**: Raw CSI data in `.dat` format  
- **Collection setup**: Intel 5300 NIC + Linux 802.11n CSI Tool  
- **CSI content**: Each `.dat` file contains a time series of complex CSI frames, typically shaped as:T × 30 × A

where:
- `T` = number of time frames  
- `30` = subcarriers  
- `A` = number of antennas (A ≥ 2)

- **Directory layout**:
dataset_raw/
body/
kyphosis/
cyc-k013.dat
normal/
abc-n001.dat
scoliosis/
zzy-s042.dat

- **File naming**:  
<identity>-<posture_code><index>.dat
- `identity`: subject code (e.g., `cyc`)  
- `posture_code`: `k` = kyphosis, `n` = normal, `s` = scoliosis  
- `index`: sample number (e.g., `013`)

---

## 🏷 Label Definitions

- **Posture**  
- Derived from the directory name  
- Classes: `kyphosis`, `normal`, `scoliosis`

- **Identity**  
- Derived from the filename prefix (before the dash)  
- Mapped to consecutive integer IDs during training

---

## ⚙️ Recommended Processing

We recommend using [CSIKit](https://github.com/Gi-z/CSIKit) for parsing `.dat` files and converting them to `.npz` format.

### Steps:
1. **Read CSI**  
 Use `IWLBeamformReader` to parse `.dat` into a `(T, 30, A)` complex array.
2. **Feature extraction**  
 - Compute complex ratio: `ratio = H_a2 / (H_a1 + ε)`  
 - Amplitude: `abs(ratio)`  
 - Phase: `angle(ratio)` (optional unwrap)  
 - Concatenate to form `(T, 60)` features
3. **Windowing**  
 Slice into fixed-length segments (e.g., length = 300, hop = 100)
4. **Save**  
 Store each segment as `.npz` with key `csi`, shape `(300, 60)`  
 Organize into `train/val/test` directories by posture

---

## 💻 Conversion Example

```python
import os
import numpy as np
from CSIKit.reader import IWLBeamformReader

def process_file(data_path, label, file, output_dir):
    csi_data_path = os.path.join(data_path, 'body')
    filename = file.split('.')[0]
    csi_file = os.path.join(csi_data_path, label, filename + '.dat')

    np.seterr(divide='ignore', invalid='ignore')

    my_reader = IWLBeamformReader()
    csi_data = my_reader.read_file(csi_file)

    csi_list = []
    for i in range(len(csi_data.frames)):
        if csi_data.frames[i].csi_matrix[:, :, :].shape[2] > 1:
            csi_list.append(csi_data.frames[i].csi_matrix[:, :, 0:2])

    csi_data = np.array(csi_list)
    csi_antenna_1 = csi_data[:, :, :, 0]
    csi_antenna_2 = csi_data[:, :, :, 1]
    csi_final = csi_antenna_2 / (csi_antenna_1 + 1e-8)

    csi_matrix_ph = np.angle(csi_final)
    csi_matrix_ph = frequency_compensate_with_polynomial_trend_removal(csi_matrix_ph)
    csi_matrix_am = np.log(np.abs(csi_final))

    y = np.concatenate((np.zeros_like(csi_matrix_am), csi_matrix_ph), axis=1)
    y[np.isinf(y)] = 0
    y[np.isnan(y)] = 0

    csi_shape = y.shape
    if y.shape[0] > 3000:
        y = y.reshape(csi_shape[0], -1)
        csi = np.stack([np.asarray(line, dtype=float)[np.newaxis, ...] for line in y], axis=0)

        c_index = 0
        for i in range(27):
            csi_feature = csi[c_index:c_index + seq_len, ...]
            np.savez(
                os.path.join(
                    output_dir,
                    data_path.split('\\')[-1] + '_' + filename + '_' + str(i) + '.npz'
                ),
                csi=csi_feature
            )
            c_index += 100
```
---

## 📥 Simple Loader Example
```python
import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class CSIDataset(Dataset):
    def __init__(self, root):
        files = sorted(glob.glob(os.path.join(root, "*", "*.npz")))
        self.samples = []
        for f in files:
            posture = os.path.basename(os.path.dirname(f))
            identity = os.path.basename(f).split("-")[0]
            self.samples.append((f, identity, posture))
        ids = [id for _, id, _ in self.samples]
        poss = [p for _, _, p in self.samples]
        self.id_map = {id: idx for idx, id in enumerate(sorted(set(ids)))}
        self.posture_map = {p: idx for idx, p in enumerate(sorted(set(poss)))}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, identity, posture = self.samples[idx]
        arr = np.load(f)["csi"].astype(np.float32)  # (T,F)
        T, F = arr.shape
        if F % 60 == 0:
            C = F // 60
            arr = arr.reshape(T, 60, C).transpose(2, 0, 1)  # (C,T,60)
        else:
            arr = arr[None, ...]
        x = torch.from_numpy(arr)
        mean = x.mean(dim=(1,2), keepdim=True)
        std = x.std(dim=(1,2), keepdim=True)
        std[std == 0] = 1
        x = (x - mean) / std
        y = {
            "identity": torch.tensor(self.id_map[identity], dtype=torch.long),
            "posture": torch.tensor(self.posture_map[posture], dtype=torch.long)
        }
        return x, y

# Example usage
if __name__ == "__main__":
    dataset = CSIDataset("dataset_npz/train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in loader:
        print(x.shape, y)
        break

