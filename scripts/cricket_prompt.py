# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import os
import jinja2
import fire
from llama import Llama

environment = jinja2.Environment()

# Set the system prompt
system_prompt_template_1 = environment.from_string("""
Cutting Knowledge Date: December 2023
Today Date: {{today_date}}

You are a helpful assistant.
""")

# User prompt template
user_input_template_1 = environment.from_string("""
You are required to answer with only JSON!!
We are going to go through some cricket commentary lines, you are supposed to figure out the batsman,
the bowler and the runs scored for that ball.
Also, we want to figure the ball type, where the ball is pitching, and where the ball is aiming to hit.

For example: 
1. Vandersay to Sundar, FOUR runs, big over, this. Another wrong'un, I think, outside off, and he picks the line well and 
reverse-sweeps fine, well to the right of Liyanage at deep third. That fielder is surely back only because 
Sri Lanka know he plays this shot -> { "batsman": "Sundar", "bowler": "Vandersay", "runs": 4, 
"ball_type": "googly/wrong'un", ""ball_pitch_area": "outside off", "ball_aim": "None"}

2. Vandersay to Dube, OUT gone, and he can't review even if he'd like to! Down the track, realises he's not to the pitch of
this flighted legbreak, and sinks onto his back knee to have a slog-sweep at it. Misses, is hit on the back thigh, 
and Ruchira Palliyaguruge gives it out immediately. Brave decision, because he's down the track on a turning pitch, 
but we'll know better when we see ball-tracking. It straightened, and it didn't turn a lot, but it still had a long way to travel.
Well, ball-tracking says that pitched on off and middle and was turning to hit a good part of middle stump -> 
{ "batsman": "Dube", "bowler": "Vandersay", "runs": 0, "ball_type": "leg break", ""ball_pitch_area": "on off and middle",
"ball_aim": "middle stump" }

3. Parag to Kamindu Mendis, no run, length ball straightening towards middle and leg, almost pops it back off the leading edge.
Lands safely though -> { "batsman": "Kamindu Mendis", "bowler": "Parag", "runs": 0, "ball_type": "straight", 
ball_pitch_area": "on length", "ball_aim": "middle and leg" }

4. Kuldeep to Kusal Mendis, FOUR runs, and Mendis continues to find the boundary. Floats this one away from off stump, and Mendis 
steps out nimbly to launch it cleanly over extra-cover -> { "batsman": "Kamindu Mendis", "bowler": "Kuldeep", "runs": 4,
"ball_type": "None", ""ball_pitch_area": "outside off stump", "ball_aim": "away from offstump" }

Please analyse the following:
{{ commentary_line }}

Only JSON.
""")

line1 = "Vandersay to Dube, no run, fullish legnbreak tuirning down leg, gets forward to defend, ends up almost deliberately padding into the leg side"
line2 = "Green to Mitchell, OUT, The short ball does the trick for Green! Banged in short outside off, Mitchell was looking to go over long-on. Does not get any power on the pull and Patidar makes no mistake at long-on."
line3 = "Siraj to Kamindu Mendis, SIX runs, full-toss, shin high, angling to middle stump from over the wicket, and Kamindu puts it away. Clears his front leg and launches it a LONG way over the wide long-on boundary"
line4 = "Yash Dayal to Gaikwad, FOUR runs, Strayed on the pads and Gaikwad cashes in. Length down leg, he shuffles across and clips fine and for four. Six fours and six already in this innings"

commentary_lines = [line1, line2, line3, line4]

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    sys_prompt = system_prompt_template_1.render(today_date="8th Aug, 24")
    chat_history = [{"role": "system", "content": sys_prompt}]
    for line in commentary_lines:
        chat_history.append({"role": "user", "content": user_input_template_1.render(commentary_line=line)})
        results = generator.chat_completion(
            [chat_history],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        print("--------------------------------------------------")
        print(f"{line} -->")
        print(results[0]['generation']['content'])
        chat_history.pop(-1)


if __name__ == "__main__":
    fire.Fire(main)
