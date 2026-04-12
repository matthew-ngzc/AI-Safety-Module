param(
    [string]$PythonExe = "python",
    [string]$Style = "",
    [int]$ReuseRewritesFromRun = 57,
    [switch]$IncludeBaselineExperiment
)

$ErrorActionPreference = "Stop"

function Invoke-Experiment {
    param(
        [string]$Name,
        [string[]]$Args
    )

    $baseArgs = @("attack_pipeline.py")
    if ($ReuseRewritesFromRun -gt 0) {
        $baseArgs += @("--reuse-rewrites-from-run", "$ReuseRewritesFromRun")
    }
    if ($Style -and $Style.Trim().Length -gt 0) {
        $baseArgs += @("--style", $Style.Trim())
    }
    $allArgs = $baseArgs + $Args

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RUNNING: $Name"
    Write-Host "COMMAND: $PythonExe $($allArgs -join ' ')"
    Write-Host "============================================================"

    $start = Get-Date
    & $PythonExe @allArgs
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start
    Write-Host ("ELAPSED: {0:n1}s" -f $elapsed.TotalSeconds)

    if ($exitCode -ne 0) {
        throw "Experiment '$Name' failed with exit code $exitCode"
    }
}

$experiments = @(
    @{
        Name = "1_gemini_25_flash_thinking_on"
        Args = @(
            "--target-provider", "gemini",
            "--target-model", "gemini-2.5-flash",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "2048",
            "--target-reasoning-effort", "none"
        )
    },
    @{
        Name = "2_claude_haiku45_thinking_on"
        Args = @(
            "--target-provider", "claude",
            "--target-model", "claude-haiku-4-5-20251001",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "2048",
            "--target-reasoning-effort", "none"
        )
    },
    @{
        Name = "3_openai_gpt5mini_reasoning_medium"
        Args = @(
            "--target-provider", "openai",
            "--target-model", "gpt-5-mini",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "0",
            "--target-reasoning-effort", "medium"
        )
    },
    @{
        Name = "4_gemini_25_flash_thinking_off"
        Args = @(
            "--target-provider", "gemini",
            "--target-model", "gemini-2.5-flash",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "0",
            "--target-reasoning-effort", "none"
        )
    },
    @{
        Name = "5_claude_haiku45_thinking_off"
        Args = @(
            "--target-provider", "claude",
            "--target-model", "claude-haiku-4-5-20251001",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "0",
            "--target-reasoning-effort", "none"
        )
    },
    @{
        Name = "6_openai_gpt5mini_reasoning_off"
        Args = @(
            "--target-provider", "openai",
            "--target-model", "gpt-5-mini",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "0",
            "--target-reasoning-effort", "none"
        )
    },
    @{
        Name = "7_openrouter_qwen3_32b_reasoning_medium"
        Args = @(
            "--target-provider", "openrouter",
            "--target-model", "qwen/qwen3-32b",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "2048",
            "--target-reasoning-effort", "medium"
        )
    },
    @{
        Name = "8_deepseek_chat_thinking_on"
        Args = @(
            "--target-provider", "deepseek",
            "--target-model", "deepseek-chat",
            "--target-max-tokens", "4096",
            "--target-thinking-budget", "2048",
            "--target-reasoning-effort", "none"
        )
    }
)

$suiteStart = Get-Date
for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    if (-not $IncludeBaselineExperiment -and $exp.Name -eq "1_gemini_25_flash_thinking_on") {
        Write-Host "Skipping baseline experiment '$($exp.Name)' (already present in source run $ReuseRewritesFromRun)."
        continue
    }
    Invoke-Experiment -Name $exp.Name -Args $exp.Args
}
$suiteElapsed = (Get-Date) - $suiteStart

Write-Host ""
Write-Host "All selected experiments completed successfully."
Write-Host ("TOTAL ELAPSED: {0:n1}s" -f $suiteElapsed.TotalSeconds)
