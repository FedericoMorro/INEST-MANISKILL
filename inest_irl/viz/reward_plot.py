import numpy as np
from inest_irl.utils.utils import is_nan_or_none


def generate_reward_plot(ax, rewards, subgoal_idxs=None, env_rewards=None, detected_subgoal_idxs=None, title="Reward", fontsizes={}):
    """Generate reward curve plot on given axis with optional subgoal and environment reward annotations."""
    
    ax.plot(rewards, linewidth=2, color='steelblue', label='Reward')
    ax.axhline(np.mean(rewards), color='blue', linestyle='-.', linewidth=2, alpha=0.5,
                label=f'Avg: {np.mean(rewards):.2f}')
    if subgoal_idxs is not None:
        # filter out np.nan values
        valid_idxs = [idx for idx in subgoal_idxs if not is_nan_or_none(idx)]
        for idx in valid_idxs:
            ax.axvline(int(idx), color='purple', linestyle=':', alpha=0.7, label='GT Subgoal(s)' if idx == valid_idxs[0] else "")
    if detected_subgoal_idxs is not None:
        # filter out np.nan values
        valid_idxs = [idx for idx in detected_subgoal_idxs if not is_nan_or_none(idx)]
        for idx in valid_idxs:
            ax.axvline(int(idx), color='green', linestyle='--', alpha=0.7, label='Detected Subgoal(s)' if idx == valid_idxs[0] else "")

    ax2 = None
    if env_rewards is not None:
        ax2 = ax.twinx()
        ax2.plot(env_rewards, linewidth=2, color='orange', label='Env Reward')
        ax2.set_ylabel('Env Reward', fontsize=fontsizes.get('label', None))
        ax2.tick_params(axis='y', labelcolor='orange')
        
    ax.set_xlabel('Step', fontsize=fontsizes.get('label', None))
    ax.set_ylabel('Reward', fontsize=fontsizes.get('label', None))
    ax.set_title(title, fontsize=fontsizes.get('title', None), fontweight='bold')
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        
    ax.legend(handles, labels, fontsize=fontsizes.get('legend', None))