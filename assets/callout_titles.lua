-- Pandoc 会把 [!NOTE] 的标题写成英文 Note。换成中文，避免读者看见英文框名。
local titles = {
  note = "提示",
  warning = "注意",
  important = "要点",
  tip = "直觉",
  caution = "小心",
}

function Div(div)
  local label = nil
  for _, class in ipairs(div.classes) do
    if titles[class] then
      label = titles[class]
      break
    end
  end
  if not label then
    return nil
  end
  for _, block in ipairs(div.content) do
    if block.t == "Div" and block.classes:includes("title") then
      block.content = { pandoc.Plain({ pandoc.Str(label) }) }
      break
    end
  end
  return div
end
