import Foundation
import PDFKit
let items=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1]))) as! [[String:Any]]
for item in items {
 let source=item["source"] as! String
 let doc=PDFDocument(url:URL(fileURLWithPath:source))!
 let page=doc.page(at:item["page"] as! Int)!
 let box=page.bounds(for:.mediaBox)
 let cut=(item["bottom_fraction"] as! Double)*box.height
 let cropped=CGRect(x:box.minX,y:box.minY+cut,width:box.width,height:box.height-cut-(item["top_fraction"] as? Double ?? 0)*box.height)
 page.setBounds(cropped,for:.cropBox);page.setBounds(cropped,for:.mediaBox)
 let result=PDFDocument();result.insert(page,at:0)
 if !result.write(to:URL(fileURLWithPath:item["dest"] as! String)){fatalError("write failed")}
}
print("Cropped existing vector pages; numerical data unchanged")
