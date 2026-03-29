param(
    [string]$BaseDir = 'D:\SAM\Rectal\CTV\146p\20260325\Slice_index'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$MethodOrder = @(
    'nnunet_all',
    'nnunet_crop',
    'nnunet_crop_SAMmask',
    'nnunet_crop_SAMbox'
)

$MethodStyles = @{
    'nnunet_all' = @{ Color = 'DodgerBlue'; Marker = 'Circle' }
    'nnunet_crop' = @{ Color = 'DarkOrange'; Marker = 'Square' }
    'nnunet_crop_SAMmask' = @{ Color = 'ForestGreen'; Marker = 'Triangle' }
    'nnunet_crop_SAMbox' = @{ Color = 'Crimson'; Marker = 'Diamond' }
}

function To-IntOrNull {
    param($Value)
    if ($null -eq $Value) { return $null }
    $s = "$Value".Trim()
    if ($s -eq '') { return $null }
    try { return [int][double]$s } catch { return $null }
}

function To-DoubleOrNull {
    param($Value)
    if ($null -eq $Value) { return $null }
    $s = "$Value".Trim()
    if ($s -eq '') { return $null }
    try { return [double]$s } catch { return $null }
}

function Get-PatientId {
    param($Row)

    if ($Row.PSObject.Properties.Name -contains 'case_id') {
        $id = To-IntOrNull $Row.case_id
        if ($null -ne $id) { return $id }
    }

    foreach ($k in @('case', 'pred_file', 'gt_case_dir')) {
        if ($Row.PSObject.Properties.Name -contains $k) {
            $v = "$($Row.$k)"
            if ($v -match '(\d+)') {
                return [int]$Matches[1]
            }
        }
    }

    throw "Cannot parse patient id from row: $($Row | ConvertTo-Json -Compress)"
}

function Normalize-Rows {
    param(
        [array]$Rows,
        [string]$Tag
    )

    $out = @()
    foreach ($row in $Rows) {
        $patientId = Get-PatientId $row

        $relLow = To-IntOrNull $row.z_relative_to_gt_lower
        $relUp = To-IntOrNull $row.z_relative_to_gt_upper

        $gtLow = To-IntOrNull $row.gt_lower_z
        $gtUp = To-IntOrNull $row.gt_upper_z
        $zAbs = To-IntOrNull $row.z
        $gtNonEmpty = To-IntOrNull $row.gt_nonempty
        if ($null -eq $gtNonEmpty) { $gtNonEmpty = 0 }

        if ($null -eq $relLow -or $null -eq $relUp) {
            if ($null -ne $zAbs -and $null -ne $gtLow -and $null -ne $gtUp) {
                $relLow = $zAbs - $gtLow
                $relUp = $zAbs - $gtUp
            } else {
                continue
            }
        }

        if ($null -ne $gtLow -and $null -ne $gtUp) {
            $upperIdx = $gtUp - $gtLow
        } else {
            $upperIdx = -1 * $relUp
        }

        $dice = To-DoubleOrNull $row.dice_2d
        $hd95 = To-DoubleOrNull $row.hd95_2d_mm

        $obj = [PSCustomObject]@{
            patient_id = $patientId
            rel_low = $relLow
            rel_up = $relUp
            cur_z = $relLow
            lower_idx = 0
            upper_idx = $upperIdx
            gt_nonempty = $gtNonEmpty
            dice = $dice
            hd95 = $hd95
        }
        $out += $obj
    }

    return $out
}

function Build-Map {
    param([array]$Rows)

    $map = @{}
    foreach ($r in $Rows) {
        $key = "{0}|{1}|{2}" -f $r.patient_id, $r.rel_low, $r.rel_up
        $map[$key] = $r
    }
    return $map
}

function New-Chart {
    param(
        [string]$Title,
        [string]$YTitle,
        [double]$XMin,
        [double]$XMax,
        [double]$XInterval,
        [double]$YMin,
        [double]$YMax,
        [double]$YInterval
    )

    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 2400
    $chart.Height = 1300
    $chart.BackColor = [System.Drawing.Color]::White

    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea 'Main'
    $area.BackColor = [System.Drawing.Color]::FromArgb(248,248,248)
    $area.AxisX.Title = 'Slice position (Lower -> Upper)'
    $area.AxisY.Title = $YTitle
    $area.AxisX.TitleFont = New-Object System.Drawing.Font('Arial', 22, [System.Drawing.FontStyle]::Bold)
    $area.AxisY.TitleFont = New-Object System.Drawing.Font('Arial', 22, [System.Drawing.FontStyle]::Bold)
    $area.AxisX.IsLabelAutoFit = $false
    $area.AxisY.IsLabelAutoFit = $false
    $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Arial', 22, [System.Drawing.FontStyle]::Regular)
    $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Arial', 22, [System.Drawing.FontStyle]::Regular)
    $area.AxisX.LabelStyle.Enabled = $true
    $area.AxisY.LabelStyle.Enabled = $true
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(220,220,220)
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(220,220,220)
    $area.AxisX.MajorTickMark.Enabled = $true
    $area.AxisY.MajorTickMark.Enabled = $true
    $area.AxisX.MajorTickMark.Size = 0.8
    $area.AxisY.MajorTickMark.Size = 0.8
    $area.AxisX.MajorTickMark.LineWidth = 1
    $area.AxisY.MajorTickMark.LineWidth = 1
    $area.AxisX.IsMarginVisible = $false
    $area.AxisX.Minimum = $XMin
    $area.AxisX.Maximum = $XMax
    $area.AxisX.Interval = $XInterval
    $area.AxisY.Minimum = $YMin
    $area.AxisY.Maximum = $YMax
    $area.AxisY.Interval = $YInterval
    $area.AxisY.LabelStyle.Format = '0.##'
    $area.Position.Auto = $false
    $area.Position.X = 4
    $area.Position.Y = 6
    $area.Position.Width = 94
    $area.Position.Height = 80
    $area.InnerPlotPosition.Auto = $false
    $area.InnerPlotPosition.X = 8
    $area.InnerPlotPosition.Y = 6
    $area.InnerPlotPosition.Width = 89
    $area.InnerPlotPosition.Height = 84

    $chart.ChartAreas.Add($area)

    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend 'Legend'
    $legend.Docking = 'Bottom'
    $legend.Alignment = 'Center'
    $legend.Font = New-Object System.Drawing.Font('Arial', 20, [System.Drawing.FontStyle]::Bold)
    $chart.Legends.Add($legend)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $Title
    $titleObj.Font = New-Object System.Drawing.Font('Arial', 28, [System.Drawing.FontStyle]::Bold)
    $titleObj.Docking = 'Top'
    $chart.Titles.Add($titleObj)

    return $chart
}

function Get-NiceInterval {
    param(
        [double]$RawInterval,
        [bool]$IsDice
    )

    if ($IsDice) {
        foreach ($c in @(0.02, 0.05, 0.1, 0.2)) {
            if ($RawInterval -le $c) { return $c }
        }
        return 0.2
    }

    foreach ($c in @(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)) {
        if ($RawInterval -le $c) { return $c }
    }
    return 100.0
}

function Get-YAxisRange {
    param(
        [double[]]$Values,
        [bool]$IsDice
    )

    if ($Values.Count -eq 0) {
        if ($IsDice) {
            return @{ Min = 0.0; Max = 1.0; Interval = 0.1 }
        }
        return @{ Min = 0.0; Max = 1.0; Interval = 0.2 }
    }

    $vmin = ($Values | Measure-Object -Minimum).Minimum
    $vmax = ($Values | Measure-Object -Maximum).Maximum
    $range = $vmax - $vmin

    if ($range -lt 1e-9) {
        if ($IsDice) {
            $pad = 0.03
        } else {
            $pad = [Math]::Max(0.5, [Math]::Abs($vmax) * 0.1)
        }
    } else {
        if ($IsDice) {
            $padFloor = 0.02
        } else {
            $padFloor = 0.2
        }
        $pad = [Math]::Max($range * 0.15, $padFloor)
    }

    $ymin = $vmin - $pad
    $ymax = $vmax + $pad

    if ($IsDice) {
        $ymin = [Math]::Max(0.0, $ymin)
        $ymax = [Math]::Min(1.0, $ymax)
        if (($ymax - $ymin) -lt 0.06) {
            $mid = ($ymax + $ymin) / 2.0
            $ymin = [Math]::Max(0.0, $mid - 0.03)
            $ymax = [Math]::Min(1.0, $mid + 0.03)
        }
    } elseif (($ymax - $ymin) -lt 0.5) {
        $mid2 = ($ymax + $ymin) / 2.0
        $ymin = $mid2 - 0.25
        $ymax = $mid2 + 0.25
    }

    $rawInterval = ($ymax - $ymin) / 6.0
    if ($rawInterval -le 0) { $rawInterval = if ($IsDice) { 0.1 } else { 0.5 } }
    $interval = Get-NiceInterval -RawInterval $rawInterval -IsDice:$IsDice

    $ymin = [Math]::Floor($ymin / $interval) * $interval
    $ymax = [Math]::Ceiling($ymax / $interval) * $interval
    if ($ymax -le $ymin) { $ymax = $ymin + $interval * 2 }

    if ($IsDice) {
        $ymin = [Math]::Max(0.0, $ymin)
        $ymax = [Math]::Min(1.0, $ymax)
        if ($ymax -le $ymin) { $ymax = [Math]::Min(1.0, $ymin + $interval * 2) }
    }

    return @{ Min = $ymin; Max = $ymax; Interval = $interval }
}

function Get-XInterval {
    param(
        [int]$XMin,
        [int]$XMax
    )
    return 4
}

function Add-LastXLabel {
    param(
        $AxisX,
        [int]$XMax,
        [int]$XMin,
        [int]$XInterval
    )

    if ((($XMax - $XMin) % $XInterval) -eq 0) {
        return
    }

    $hasLabel = $false
    foreach ($cl in $AxisX.CustomLabels) {
        if ($cl.Text -eq ([string]$XMax)) {
            $hasLabel = $true
            break
        }
    }
    if (-not $hasLabel) {
        $lbl = New-Object System.Windows.Forms.DataVisualization.Charting.CustomLabel
        $lbl.FromPosition = $XMax - 0.45
        $lbl.ToPosition = $XMax + 0.45
        $lbl.Text = [string]$XMax
        $AxisX.CustomLabels.Add($lbl)
    }
}

function Set-XLabels {
    param(
        $AxisX,
        [int]$XMin,
        [int]$XMax,
        [int]$XInterval
    )

    $AxisX.CustomLabels.Clear()

    $positions = New-Object System.Collections.Generic.List[int]
    for ($x = $XMin; $x -le $XMax; $x += $XInterval) {
        $positions.Add($x)
    }
    if ($positions.Count -eq 0 -or $positions[$positions.Count - 1] -ne $XMax) {
        $positions.Add($XMax)
    }

    foreach ($pos in $positions) {
        $lbl = New-Object System.Windows.Forms.DataVisualization.Charting.CustomLabel
        $lbl.FromPosition = $pos - 0.45
        $lbl.ToPosition = $pos + 0.45
        $lbl.Text = [string]$pos
        $AxisX.CustomLabels.Add($lbl)
    }
}

$paths = @{
    'nnunet_all' = Join-Path $BaseDir 'slice_nnunet_all.csv'
    'nnunet_crop' = Join-Path $BaseDir 'slice_nnunet_crop.csv'
    'nnunet_crop_SAMmask' = Join-Path $BaseDir 'slice_nnunet_crop_SAMmask.csv'
    'nnunet_crop_SAMbox' = Join-Path $BaseDir 'slice_nnunet_crop_SAMbox.csv'
}

foreach ($k in $paths.Keys) {
    if (-not (Test-Path -LiteralPath $paths[$k])) {
        throw "Missing input CSV: $($paths[$k])"
    }
}

$data = @{}
foreach ($m in $MethodOrder) {
    $rows = Import-Csv -Path $paths[$m]
    $norm = Normalize-Rows -Rows $rows -Tag $m
    $map = Build-Map -Rows $norm
    $data[$m] = $map
    Write-Host "[Info] $m rows=$($rows.Count) normalized=$($norm.Count) keys=$($map.Count)"
}

$gtCrop = New-Object 'System.Collections.Generic.HashSet[string]'
$gtMask = New-Object 'System.Collections.Generic.HashSet[string]'
$gtBox = New-Object 'System.Collections.Generic.HashSet[string]'

foreach ($kv in $data['nnunet_crop'].GetEnumerator()) {
    if ([int]$kv.Value.gt_nonempty -eq 1) { [void]$gtCrop.Add($kv.Key) }
}
foreach ($kv in $data['nnunet_crop_SAMmask'].GetEnumerator()) {
    if ([int]$kv.Value.gt_nonempty -eq 1) { [void]$gtMask.Add($kv.Key) }
}
foreach ($kv in $data['nnunet_crop_SAMbox'].GetEnumerator()) {
    if ([int]$kv.Value.gt_nonempty -eq 1) { [void]$gtBox.Add($kv.Key) }
}

Write-Host "[Info] GT keys nnunet_crop = $($gtCrop.Count)"
Write-Host "[Info] GT keys nnunet_crop_SAMmask = $($gtMask.Count)"
Write-Host "[Info] GT keys nnunet_crop_SAMbox = $($gtBox.Count)"

$keep = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($k in $gtMask) {
    if ($gtCrop.Contains($k) -and $gtBox.Contains($k)) {
        [void]$keep.Add($k)
    }
}
Write-Host "[Info] keep keys = $($keep.Count)"

$patientMap = @{}
foreach ($k in $keep) {
    $parts = $k -split '\|'
    $patientId = [int]$parts[0]
    if (-not $patientMap.ContainsKey($patientId)) {
        $patientMap[$patientId] = New-Object System.Collections.ArrayList
    }
    [void]$patientMap[$patientId].Add($k)
}

$diceDir = Join-Path $BaseDir 'Dice_2d'
$hdDir = Join-Path $BaseDir 'HD95_2d'
New-Item -ItemType Directory -Force -Path $diceDir | Out-Null
New-Item -ItemType Directory -Force -Path $hdDir | Out-Null

$patientIds = $patientMap.Keys | Sort-Object
$total = $patientIds.Count
$idx = 0

foreach ($patientId in $patientIds) {
    $idx++
    $keys = @($patientMap[$patientId])

    $rowsForX = @()
    foreach ($k in $keys) {
        $ref = $data['nnunet_crop'][$k]
        if ($null -eq $ref) { $ref = $data['nnunet_crop_SAMmask'][$k] }
        if ($null -eq $ref) { $ref = $data['nnunet_crop_SAMbox'][$k] }
        if ($null -ne $ref) { $rowsForX += $ref }
    }
    if ($rowsForX.Count -eq 0) { continue }

    $xVals = $rowsForX | ForEach-Object { [int]$_.cur_z } | Sort-Object
    $xMin = [int]($xVals | Measure-Object -Minimum).Minimum
    $xMax = [int]($xVals | Measure-Object -Maximum).Maximum
    if ($xMax -le $xMin) { $xMax = $xMin + 1 }
    $xInterval = Get-XInterval -XMin $xMin -XMax $xMax

    foreach ($metric in @('dice', 'hd95')) {
        $isDice = $metric -eq 'dice'

        $allY = @()
        foreach ($m in $MethodOrder) {
            foreach ($k in $keys) {
                if ($data[$m].ContainsKey($k)) {
                    $v = $data[$m][$k].$metric
                    if ($null -ne $v) { $allY += [double]$v }
                }
            }
        }

        $yr = Get-YAxisRange -Values $allY -IsDice:$isDice

        $titleMetric = if ($isDice) { 'Dice' } else { 'HD95' }
        $yTitle = if ($isDice) { 'Dice coefficient' } else { 'HD95 (mm)' }
        $caseName = ('p_{0:D2}' -f $patientId)

        $chart = New-Chart -Title "Slice-wise $titleMetric Curve (Case $caseName)" -YTitle $yTitle -XMin $xMin -XMax $xMax -XInterval $xInterval -YMin $yr.Min -YMax $yr.Max -YInterval $yr.Interval
        $axisX = $chart.ChartAreas['Main'].AxisX
        $axisX.LabelStyle.IsEndLabelVisible = $true
        Set-XLabels -AxisX $axisX -XMin $xMin -XMax $xMax -XInterval $xInterval

        foreach ($m in $MethodOrder) {
            $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $m
            $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
            $series.BorderWidth = 4
            $series.MarkerSize = 10
            $series.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::$($MethodStyles[$m].Marker)
            $series.Color = [System.Drawing.Color]::FromName($MethodStyles[$m].Color)
            $series.XValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Int32
            $series.YValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Double

            $pts = @()
            foreach ($k in $keys) {
                if ($data[$m].ContainsKey($k)) {
                    $r = $data[$m][$k]
                    $v = $r.$metric
                    if ($null -ne $v) {
                        $pts += [PSCustomObject]@{ x = [int]$r.cur_z; y = [double]$v }
                    }
                }
            }
            $pts = $pts | Sort-Object x

            foreach ($p in $pts) {
                [void]$series.Points.AddXY($p.x, $p.y)
            }

            $chart.Series.Add($series)
        }

        $outDir = if ($isDice) { $diceDir } else { $hdDir }
        if ($isDice) {
            $suffix = 'dice_2d'
        } else {
            $suffix = 'hd95_2d'
        }
        $outPath = Join-Path $outDir ("{0}_{1}.png" -f $caseName, $suffix)

        $chart.SaveImage($outPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
        $chart.Dispose()
    }

    Write-Host "[Progress] $idx/$total done: p_$('{0:D2}' -f $patientId)"
}

Write-Host "[Done] Dice plots: $diceDir"
Write-Host "[Done] HD95 plots: $hdDir"
