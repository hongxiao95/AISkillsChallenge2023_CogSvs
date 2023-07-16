﻿#pragma checksum "D:\CustomVisionDemo\CustomVisionDemo\MainPage.xaml" "{406ea660-64cf-4c82-b6f0-42d48172a799}" "BD5D6B67AEBFC5BD6B0E7A79F02713D5"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace CustomVisionDemo
{
    partial class MainPage : 
        global::Windows.UI.Xaml.Controls.Page, 
        global::Windows.UI.Xaml.Markup.IComponentConnector,
        global::Windows.UI.Xaml.Markup.IComponentConnector2
    {
        /// <summary>
        /// Connect()
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.18362.1")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public void Connect(int connectionId, object target)
        {
            switch(connectionId)
            {
            case 2: // MainPage.xaml line 25
                {
                    this.pageTitle = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 3: // MainPage.xaml line 28
                {
                    this.ContentRoot = (global::Windows.UI.Xaml.Controls.StackPanel)(target);
                }
                break;
            case 4: // MainPage.xaml line 35
                {
                    this.ContentRootActions = (global::Windows.UI.Xaml.Controls.StackPanel)(target);
                }
                break;
            case 5: // MainPage.xaml line 70
                {
                    this.status = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 6: // MainPage.xaml line 62
                {
                    this.playbackCanvas3 = (global::Windows.UI.Xaml.Controls.Canvas)(target);
                }
                break;
            case 7: // MainPage.xaml line 63
                {
                    this.playbackElement3 = (global::Windows.UI.Xaml.Controls.MediaElement)(target);
                }
                break;
            case 8: // MainPage.xaml line 55
                {
                    this.VideoTitle = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 9: // MainPage.xaml line 56
                {
                    this.VideoCanvas = (global::Windows.UI.Xaml.Controls.Canvas)(target);
                }
                break;
            case 10: // MainPage.xaml line 57
                {
                    this.playbackElement = (global::Windows.UI.Xaml.Controls.MediaElement)(target);
                }
                break;
            case 11: // MainPage.xaml line 49
                {
                    this.ImageTitle = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 12: // MainPage.xaml line 50
                {
                    this.PhotoCanvas = (global::Windows.UI.Xaml.Controls.Canvas)(target);
                }
                break;
            case 13: // MainPage.xaml line 51
                {
                    this.captureImage = (global::Windows.UI.Xaml.Controls.Image)(target);
                }
                break;
            case 14: // MainPage.xaml line 43
                {
                    this.PreviewTitle = (global::Windows.UI.Xaml.Controls.TextBlock)(target);
                }
                break;
            case 15: // MainPage.xaml line 44
                {
                    this.PreviewCanvas = (global::Windows.UI.Xaml.Controls.Canvas)(target);
                }
                break;
            case 16: // MainPage.xaml line 45
                {
                    this.previewElement = (global::Windows.UI.Xaml.Controls.CaptureElement)(target);
                }
                break;
            case 17: // MainPage.xaml line 36
                {
                    this.takePhoto = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.takePhoto).Click += this.takePhoto_Click;
                }
                break;
            case 18: // MainPage.xaml line 37
                {
                    this.recordVideo = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.recordVideo).Click += this.recordVideo_Click;
                }
                break;
            case 19: // MainPage.xaml line 38
                {
                    this.recordAudio = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.recordAudio).Click += this.recordAudio_Click;
                }
                break;
            case 20: // MainPage.xaml line 29
                {
                    this.video_init = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.video_init).Click += this.initVideo_Click;
                }
                break;
            case 21: // MainPage.xaml line 30
                {
                    this.audio_init = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.audio_init).Click += this.initAudioOnly_Click;
                }
                break;
            case 22: // MainPage.xaml line 31
                {
                    this.cleanup = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.cleanup).Click += this.cleanup_Click;
                }
                break;
            case 23: // MainPage.xaml line 32
                {
                    this.customvision = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.customvision).Click += this.customvision_Click;
                }
                break;
            default:
                break;
            }
            this._contentLoaded = true;
        }

        /// <summary>
        /// GetBindingConnector(int connectionId, object target)
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.18362.1")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::Windows.UI.Xaml.Markup.IComponentConnector GetBindingConnector(int connectionId, object target)
        {
            global::Windows.UI.Xaml.Markup.IComponentConnector returnValue = null;
            return returnValue;
        }
    }
}

