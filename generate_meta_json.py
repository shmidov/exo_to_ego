import numpy as np
import pandas as pd
import json


if __name__ == "__main__":
    take_name = "example_video"
    exo_path = f"data/videos/{take_name}/exo.mp4"
    ego_prior_path = f"data/videos/{take_name}/ego_Prior.mp4"
    prompt = "[Exo view]\n**Scene Overview:** The scene takes place in an indoor workspace resembling a classroom or laboratory environment. A long wooden table occupies the center of the frame, surrounded by multiple blue rolling office chairs, most of which are unoccupied. The floor is carpeted in a neutral pattern, and the room appears well-lit by overhead lighting. On the table in the foreground are several objects: a white rectangular box with printed graphics, a pair of blue-handled scissors, loose instruction sheets, and small components laid out near the woman. The woman is seated on a chair at the table, facing slightly to the right of the camera. She is wearing a light-colored jacket and dark pants, and she has dark hair tied back. She is also wearing a pair of smart glasses or a head-mounted device, suggesting an augmented or assisted task. A tripod-mounted camera is visible in the background, reinforcing the impression of a recorded or experimental setup.\n**Action Analysis:** In the first frame, the woman is seated upright at the table, holding a sheet of paper with both hands and looking down at it attentively. Her posture is slightly leaned forward, indicating concentration. The surrounding objects on the table appear organized for a task involving assembly or setup. As the sequence progresses to the second frame, she begins manipulating the paper, adjusting its orientation as if reading or following instructions. Her hands move closer together, and her gaze remains fixed on the document. In the third frame, she shifts her attention briefly toward the items on the table, aligning the instruction sheet with the box and small components. Her body remains mostly stationary, with subtle hand movements indicating careful, step-by-step execution. By the fourth frame, she continues the task, holding the paper closer to her body while preparing to interact with the objects on the table. The overall action flow suggests a deliberate and methodical process, focused on understanding instructions and preparing for or executing a precise manual task.\n[Ego view]\n**Scene Overview:** From the first-person perspective of the woman, the view is directed downward toward the tabletop. Her hands and forearms are visible in the foreground, holding a printed instruction sheet. The wooden surface of the table fills most of the visual field. Directly ahead are the white box, loose papers, and small components, while the blue-handled scissors rest slightly to the left. The edges of nearby chairs and the room’s open space are visible in peripheral vision. The smart glasses frame the scene from eye level, emphasizing the task space directly in front of her.\n**Action Analysis:** In the first frame, her hands hold the instruction sheet steady while she reads, with her gaze focused on the text and diagrams. The proximity of the paper to her eyes suggests careful attention to detail. As the second frame unfolds, her hands adjust the paper slightly, rotating or repositioning it to better view a specific section. The motion is controlled and minimal, indicating precision rather than haste. In the third frame, her focus alternates between the instruction sheet and the objects on the table, as if mentally mapping the written steps to the physical components. Her hands hover momentarily above the table, ready to proceed. By the fourth frame, she brings the paper closer again, preparing for the next step in the task. The egocentric sequence highlights a focused, hands-on interaction with instructional material and tools, emphasizing careful planning and execution from the woman’s own viewpoint."
    camera_extrinsics = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    ego_intrinsics = [[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]]

    vipe_intrinsics = np.load(f"vipe_results/{take_name}/intrinsics/exo.npz")
    vipe_intrinsics_data = vipe_intrinsics["data"][0, :].astype(float)
    camera_intrinsics = [[vipe_intrinsics_data[0], 0.0, vipe_intrinsics_data[2]], 
                         [0.0, vipe_intrinsics_data[1], vipe_intrinsics_data[3]], 
                         [0.0, 0.0, 1.0]]

    ego_extrinsics = [[[  0.2950,   0.9162,   0.2713,  -0.3183],
                       [ -0.8204,   0.0974,   0.5634,  -0.2579],
                       [  0.4897,  -0.3888,   0.7804,  -1.5092]]] * vipe_intrinsics["data"].shape[0]

    meta_json = {"test_datasets": 
        [{
            "exo_path": exo_path,
            "ego_prior_path": ego_prior_path,
            "prompt": prompt,
            "camera_extrinsics": camera_extrinsics,
            "camera_intrinsics": camera_intrinsics,
            "ego_intrinsics": ego_intrinsics,
            "ego_extrinsics": ego_extrinsics
        }]
    }

    with open("data/videos/meta.json", "w") as f:
        json.dump(meta_json, f, indent=4)